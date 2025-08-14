# /data/boltz_webui/lead_optimization/optimization_engine_new.py

"""
Advanced optimization engine for drug discovery
Uses MMPDB and Boltz-WebUI to optimize lead compounds
"""

import os
import json
import time
import logging
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from rdkit import Chem

from config import OptimizationConfig
from api_client import BoltzOptimizationClient
from mmp_engine import MMPEngine
from scoring_system import MultiObjectiveScoring, CompoundScore
from exceptions import OptimizationError, InvalidCompoundError

logger = logging.getLogger(__name__)

@dataclass
@dataclass
class OptimizationResult:
    """Results from optimization run"""
    original_compound: str
    strategy: str
    candidates: List[Dict[str, Any]]
    scores: List[CompoundScore]
    top_candidates: List[Dict[str, Any]]
    execution_time: float
    statistics: Dict[str, Any] = None
    total_candidates: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}

@dataclass
class OptimizationCandidate:
    """Individual optimization candidate"""
    smiles: str
    compound_id: str
    mmp_transformation: str
    prediction_results: Dict[str, Any] = None
    scores: CompoundScore = None

class OptimizationEngine:
    """
    Lead optimization engine using MMPDB + Boltz-WebUI
    Similar architecture to virtual_screening for consistency
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Initialize components
        self.boltz_client = BoltzOptimizationClient(config.boltz_api)
        self.mmp_engine = MMPEngine(config.mmp_database)
        self.scoring_system = MultiObjectiveScoring(config.scoring_weights, config.filters)
        
        # Results storage
        self.optimization_results: List[CompoundScore] = []
        
        # Global candidate counter for unique IDs across generations
        self.candidate_counter = 0
        
        # Setup directories
        config.setup_directories()
        
        logger.info("Lead optimization engine initialized")
    
    def optimize_compound(self, 
                         compound_smiles: str,
                         target_protein_yaml: str,
                         strategy: str = "scaffold_hopping",
                         max_candidates: int = 50,
                         output_dir: Optional[str] = None,
                         iterations: int = 1,
                         batch_size: int = 4,
                         top_k_per_iteration: int = 5) -> OptimizationResult:
        """
        Optimize a single compound using MMPDB + Boltz-WebUI with iterative evolution
        
        Args:
            compound_smiles: Input compound SMILES
            target_protein_yaml: Target protein configuration
            strategy: Optimization strategy
            max_candidates: Maximum number of candidates to generate per iteration
            output_dir: Output directory for results
            iterations: Number of optimization iterations (genetic evolution)
            batch_size: Number of compounds to submit to Boltz simultaneously (GPU limit)
            top_k_per_iteration: Top compounds to use as seeds for next iteration
            
        Returns:
            OptimizationResult object
        """
        start_time = time.time()
        
        try:
            # Validate input compound
            if not self._is_valid_smiles(compound_smiles):
                raise InvalidCompoundError(f"Invalid compound SMILES: {compound_smiles}")
            
            # Setup output directory
            if not output_dir:
                output_dir = os.path.join(self.config.temp_dir, f"optimization_{int(time.time())}")
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info("🚀 Starting iterative optimization")
            logger.info(f"Input compound: {compound_smiles[:50]}...")
            logger.info(f"Strategy: {strategy}")
            logger.info(f"Iterations: {iterations}")
            logger.info(f"Max candidates per iteration: {max_candidates}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Top-K per iteration: {top_k_per_iteration}")
            
            # Initialize batch evaluator and evolution engine
            from .batch_evaluation import BatchEvaluator
            from .molecular_evolution import MolecularEvolutionEngine
            
            batch_evaluator = BatchEvaluator(self.boltz_client, self.scoring_system, batch_size)
            evolution_engine = MolecularEvolutionEngine()
            
            # Track all evaluated candidates across iterations
            all_evaluated_candidates = []
            generation_results = []  # Track results per generation
            current_seeds = [compound_smiles]  # Start with original compound
            
            # Iterative optimization loop with evolution
            for iteration in range(iterations):
                logger.info(f"🧬 === 进化第 {iteration + 1}/{iterations} 代 ===")
                
                # Get adaptive parameters for this generation
                convergence_rate = evolution_engine.assess_population_diversity(all_evaluated_candidates) if all_evaluated_candidates else 1.0
                adaptive_params = evolution_engine.calculate_adaptive_parameters(iteration, iterations, convergence_rate)
                
                # Generate candidates using evolution-guided strategies
                iteration_candidates = []
                
                if iteration == 0:
                    # First generation: explore around original compound
                    logger.info(f"🌱 第一代：从原始化合物生成候选")
                    candidates = self._generate_candidates_with_mmpdb(compound_smiles, strategy, max_candidates)
                    if candidates:
                        iteration_candidates.extend(candidates)
                else:
                    # Subsequent generations: use evolution engine
                    logger.info(f"🧬 第{iteration + 1}代：使用进化策略")
                    
                    if len(all_evaluated_candidates) >= top_k_per_iteration:
                        # Select parents using evolution engine
                        elite_size = max(1, top_k_per_iteration // 2)
                        diversity_size = max(1, top_k_per_iteration - elite_size)
                        
                        elite_compounds, diverse_compounds = evolution_engine.select_parents_for_next_generation(
                            all_evaluated_candidates, top_k_per_iteration, elite_size
                        )
                        
                        # Generate next generation strategies
                        strategies = evolution_engine.generate_next_generation_strategies(elite_compounds, diverse_compounds)
                        
                        # Generate candidates based on evolution strategies
                        for parent_smiles, evo_strategy, weight in strategies[:max_candidates]:
                            logger.info(f"  从 {parent_smiles[:30]}... 使用策略 {evo_strategy}")
                            candidates = self._generate_candidates_with_mmpdb(parent_smiles, evo_strategy, max(1, int(max_candidates * weight)))
                            if candidates:
                                iteration_candidates.extend(candidates[:int(max_candidates * weight)])
                    else:
                        # Not enough candidates yet, continue with current seeds
                        for seed_compound in current_seeds:
                            candidates = self._generate_candidates_with_mmpdb(seed_compound, strategy, max_candidates // len(current_seeds))
                            if candidates:
                                iteration_candidates.extend(candidates)
                
                if not iteration_candidates:
                    logger.warning(f"第 {iteration + 1} 代没有生成候选化合物，停止进化")
                    break
                
                # Remove duplicates and previously evaluated compounds
                seen_smiles = {c.smiles for c in all_evaluated_candidates}
                unique_candidates = [c for c in iteration_candidates if c.smiles not in seen_smiles]
                
                logger.info(f"第 {iteration + 1} 代总共生成 {len(iteration_candidates)} 个候选，去重后剩余 {len(unique_candidates)} 个")
                
                if not unique_candidates:
                    logger.warning(f"第 {iteration + 1} 代没有新的候选化合物，停止进化")
                    break
                
                # Evaluate candidates using batch processing
                evaluated_candidates = batch_evaluator.evaluate_candidates_batch(
                    unique_candidates,
                    target_protein_yaml,
                    output_dir,
                    compound_smiles
                )
                
                all_evaluated_candidates.extend(evaluated_candidates)
                generation_results.append(evaluated_candidates.copy())
                
                logger.info(f"✅ 第 {iteration + 1} 代完成，成功评估 {len(evaluated_candidates)} 个候选化合物")
                
                # Check for early termination using evolution engine
                if iteration > 0 and evolution_engine.suggest_termination(generation_results, patience=2):
                    logger.info("🏁 进化提前收敛，停止优化")
                    break
                
                # Select seeds for next generation (if not last iteration)
                if iteration < iterations - 1 and evaluated_candidates:
                    # Use evolution engine for better selection
                    if len(all_evaluated_candidates) >= top_k_per_iteration:
                        elite_compounds, diverse_compounds = evolution_engine.select_parents_for_next_generation(
                            all_evaluated_candidates, top_k_per_iteration, top_k_per_iteration // 2
                        )
                        selected_compounds = elite_compounds + diverse_compounds
                    else:
                        # Fallback to simple top-k selection
                        sorted_candidates = sorted(all_evaluated_candidates, 
                                                key=lambda x: x.scores.combined_score if x.scores else 0, 
                                                reverse=True)
                        selected_compounds = sorted_candidates[:top_k_per_iteration]
                    
                    current_seeds = [c.smiles for c in selected_compounds]
                    
                    logger.info(f"🎯 选择第 {iteration + 2} 代的 {len(current_seeds)} 个种子:")
                    for i, candidate in enumerate(selected_compounds):
                        score = candidate.scores.combined_score if candidate.scores else 0
                        logger.info(f"  {i+1}. {candidate.compound_id} - 分数: {score:.4f}")
                        
                    # Log population diversity
                    diversity = evolution_engine.assess_population_diversity(all_evaluated_candidates)
                    logger.info(f"📊 当前种群多样性: {diversity:.3f}")
            
            # Generate final results
            logger.info(f"🎉 优化完成！总共评估了 {len(all_evaluated_candidates)} 个候选化合物")
            
            if not all_evaluated_candidates:
                logger.warning("No candidates were successfully evaluated")
                return self._create_empty_result(compound_smiles, strategy, start_time)
            
            # Sort all candidates by score
            sorted_candidates = sorted(all_evaluated_candidates, 
                                    key=lambda x: x.scores.combined_score if x.scores else 0, 
                                    reverse=True)
            
            # Create optimization result
            execution_time = time.time() - start_time
            result = OptimizationResult(
                original_compound=compound_smiles,
                strategy=strategy,
                candidates=[self._candidate_to_dict(c) for c in sorted_candidates],
                scores=[c.scores for c in sorted_candidates if c.scores],
                top_candidates=[self._candidate_to_dict(c) for c in sorted_candidates[:10]],
                execution_time=execution_time,
                total_candidates=len(sorted_candidates),
                success_rate=len([c for c in all_evaluated_candidates if c.scores]) / len(all_evaluated_candidates) if all_evaluated_candidates else 0
            )
            
            # Save summary (no final CSV needed - real-time CSV already exists)
            self._save_optimization_summary(result, output_dir)
            
            logger.info(f"Optimization completed in {execution_time:.2f}s")
            logger.info(f"Success rate: {result.success_rate:.1%}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}")
    
    def _generate_candidates_with_mmpdb(self, 
                                       compound_smiles: str, 
                                       strategy: str, 
                                       max_candidates: int) -> List[OptimizationCandidate]:
        """Generate candidate compounds using MMPDB"""
        try:
            logger.info(f"Generating candidates with MMPDB, strategy: {strategy}")
            
            if strategy == "scaffold_hopping":
                mmp_results = self.mmp_engine.scaffold_hopping(compound_smiles, max_candidates * 2)  # Generate more, then select
            elif strategy == "fragment_replacement":
                mmp_results = self.mmp_engine.fragment_replacement(compound_smiles, max_candidates * 2)
            else:
                # Default to scaffold hopping
                mmp_results = self.mmp_engine.scaffold_hopping(compound_smiles, max_candidates * 2)
            
            logger.info(f"MMPDB generated {len(mmp_results)} raw candidates")
            
            # Sort by similarity/quality and take top candidates
            if len(mmp_results) > max_candidates:
                # Sort by similarity score (descending) - higher similarity is often better for optimization
                sorted_results = sorted(mmp_results, key=lambda x: x.get('similarity', 0), reverse=True)
                mmp_results = sorted_results[:max_candidates]
                logger.info(f"Selected top {max_candidates} candidates based on similarity")
            
            candidates = []
            for i, result in enumerate(mmp_results):
                self.candidate_counter += 1
                candidate = OptimizationCandidate(
                    smiles=result['smiles'],
                    compound_id=f"cand_{self.candidate_counter:04d}",
                    mmp_transformation=result.get('transformation_rule', result.get('transformation_description', 'unknown'))
                )
                candidates.append(candidate)
            
            logger.info(f"Generated {len(candidates)} candidates from MMPDB")
            return candidates
            
        except Exception as e:
            logger.error(f"MMPDB candidate generation failed: {e}")
            return []
    
    def _evaluate_candidates_with_boltz(self, 
                                       candidates: List[OptimizationCandidate],
                                       target_protein_yaml: str,
                                       output_dir: str,
                                       original_compound: str = None,
                                       batch_size: int = 4) -> List[OptimizationCandidate]:
        """Evaluate candidates using Boltz-WebUI with batch processing and real-time CSV updates"""
        evaluated_candidates = []
        
        # Read target protein configuration
        with open(target_protein_yaml, 'r') as f:
            target_config = yaml.safe_load(f)
        
        # Create temp directory for configs
        temp_config_dir = os.path.join(output_dir, "temp_configs")
        os.makedirs(temp_config_dir, exist_ok=True)
        
        # Set current output dir for debug YAML saving
        self._current_output_dir = output_dir
        
        # Initialize real-time CSV file
        csv_file = os.path.join(output_dir, "optimization_results.csv")
        self._initialize_csv_file(csv_file)
        
        logger.info(f"Evaluating {len(candidates)} candidates with Boltz-WebUI")
        logger.info(f"Real-time results will be saved to: {csv_file}")
        
        for i, candidate in enumerate(candidates):
            try:
                logger.info(f"Processing candidate {i+1}/{len(candidates)}: {candidate.compound_id}")
                
                # Create config for this candidate
                config_yaml = self._create_candidate_config_yaml(
                    candidate, target_config
                )
                
                if not config_yaml:
                    logger.warning(f"Failed to create config for {candidate.compound_id}")
                    self._write_csv_row(csv_file, candidate, original_compound, status="config_failed")
                    continue
                
                # Submit to Boltz-WebUI
                task_id = self.boltz_client.submit_optimization_job(
                    yaml_content=config_yaml,
                    job_name=f"opt_{candidate.compound_id}",
                    compound_smiles=candidate.smiles
                )
                
                if not task_id:
                    logger.warning(f"Failed to submit {candidate.compound_id}")
                    self._write_csv_row(csv_file, candidate, original_compound, status="submit_failed")
                    continue
                
                logger.info(f"Submitted {candidate.compound_id} as task {task_id}")
                
                # Wait for completion
                result = self.boltz_client.poll_job_status(task_id)
                
                if result and result.get('status') == 'completed':
                    # Download results
                    result_dir = os.path.join(output_dir, "results", candidate.compound_id)
                    try:
                        result_files = self.boltz_client.download_results(task_id, result_dir)
                        if result_files:  # Check if download was successful
                            # Parse prediction results
                            prediction_results = self._parse_prediction_results(result_dir)
                            candidate.prediction_results = prediction_results
                            
                            # Score the candidate immediately
                            try:
                                score = self.scoring_system.score_compound(
                                    smiles=candidate.smiles,
                                    boltz_results=candidate.prediction_results,
                                    reference_smiles=original_compound
                                )
                                candidate.scores = score
                                
                                # Write to CSV immediately
                                self._write_csv_row(csv_file, candidate, original_compound, 
                                                  status="completed", score=score, task_id=task_id)
                                
                                evaluated_candidates.append(candidate)
                                logger.info(f"✅ {candidate.compound_id} completed - Score: {score.combined_score:.4f}")
                                
                            except Exception as e:
                                logger.error(f"Scoring failed for {candidate.compound_id}: {e}")
                                self._write_csv_row(csv_file, candidate, original_compound, 
                                                  status="scoring_failed", task_id=task_id)
                        else:
                            logger.warning(f"Failed to download results for {candidate.compound_id}")
                            self._write_csv_row(csv_file, candidate, original_compound, 
                                              status="download_failed", task_id=task_id)
                    except Exception as e:
                        logger.warning(f"Error downloading results for {candidate.compound_id}: {e}")
                        self._write_csv_row(csv_file, candidate, original_compound, 
                                          status="download_error", task_id=task_id)
                else:
                    logger.warning(f"Task {task_id} failed or timed out")
                    failure_reason = result.get('error', 'unknown_error') if result else 'timeout'
                    self._write_csv_row(csv_file, candidate, original_compound, 
                                      status=f"task_failed_{failure_reason}", task_id=task_id)
                    
            except Exception as e:
                logger.error(f"Error evaluating {candidate.compound_id}: {e}")
                self._write_csv_row(csv_file, candidate, original_compound, status="error")
        
        logger.info(f"Successfully evaluated {len(evaluated_candidates)} candidates")
        logger.info(f"Real-time results saved in: {csv_file}")
        return evaluated_candidates
    
    def _create_candidate_config_yaml(self, 
                                      candidate: OptimizationCandidate,
                                      target_config: Dict) -> Optional[str]:
        """Create YAML config content for candidate"""
        try:
            import copy
            config = copy.deepcopy(target_config)
            
            # Add the candidate compound as a ligand
            ligand_id = "L"  # Use simple letter ID as required by Boltz
            ligand_entry = {
                "ligand": {
                    "id": ligand_id,
                    "smiles": candidate.smiles
                }
            }
            
            # Add to sequences
            if 'sequences' not in config:
                config['sequences'] = []
            config['sequences'].append(ligand_entry)
            
            # Clear any existing affinity properties and add only our candidate
            # This ensures we only have ONE affinity ligand (Boltz requirement)
            config['properties'] = [{
                "affinity": {
                    "binder": ligand_id
                }
            }]
            
            # Convert to YAML string
            import yaml
            yaml_content = yaml.dump(config, default_flow_style=False, allow_unicode=True)
            
            # Save YAML for debugging (use output_dir parameter from _evaluate_candidates_with_boltz)
            if hasattr(self, '_current_output_dir'):
                debug_yaml_path = os.path.join(self._current_output_dir, "temp_configs", f"{candidate.compound_id}_config.yaml")
                os.makedirs(os.path.dirname(debug_yaml_path), exist_ok=True)
                with open(debug_yaml_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)
                logger.debug(f"Saved debug YAML to {debug_yaml_path}")
            
            return yaml_content
            
        except Exception as e:
            logger.error(f"Failed to create config for {candidate.compound_id}: {e}")
            return None
    
    def _initialize_csv_file(self, csv_file: str):
        """Initialize CSV file with headers"""
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'compound_id', 
                'original_smiles',
                'optimized_smiles',
                'mmp_transformation',
                'status',
                'task_id',
                'combined_score',
                'binding_affinity',
                'drug_likeness', 
                'synthetic_accessibility',
                'novelty',
                'stability',
                'plddt',
                'iptm',
                'binding_probability',
                'ic50_um',
                'molecular_weight',
                'logp',
                'lipinski_violations',
                'qed_score'
            ])
    
    def _write_csv_row(self, csv_file: str, candidate: OptimizationCandidate, 
                       original_compound: str = None, status: str = "processing", 
                       score: 'CompoundScore' = None, task_id: str = None):
        """Write a candidate result to CSV file immediately"""
        import csv
        import time
        
        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Prepare row data
                row = [
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    candidate.compound_id,
                    original_compound or '',
                    candidate.smiles,
                    candidate.mmp_transformation,
                    status,
                    task_id or '',
                ]
                
                if score:
                    row.extend([
                        f"{score.combined_score:.4f}",
                        f"{score.binding_affinity:.4f}",
                        f"{score.drug_likeness:.4f}",
                        f"{score.synthetic_accessibility:.4f}", 
                        f"{score.novelty:.4f}",
                        f"{score.stability:.4f}",
                        f"{score.plddt:.4f}",
                        f"{score.iptm:.4f}",
                        f"{score.binding_probability:.4f}",
                        f"{score.ic50_um:.4f}",
                        f"{score.properties.get('molecular_weight', 0):.2f}",
                        f"{score.properties.get('logp', 0):.2f}",
                        score.lipinski_violations,
                        f"{score.qed_score:.4f}"
                    ])
                else:
                    # Empty values for failed/incomplete candidates
                    row.extend([''] * 14)
                
                writer.writerow(row)
                
            logger.debug(f"Written CSV row for {candidate.compound_id}: {status}")
            
        except Exception as e:
            logger.error(f"Failed to write CSV row for {candidate.compound_id}: {e}")
    
    def _parse_prediction_results(self, result_dir: str) -> Dict[str, Any]:
        """Parse Boltz prediction results"""
        results = {}
        
        # Parse affinity data
        affinity_file = os.path.join(result_dir, "affinity_data.json")
        if os.path.exists(affinity_file):
            try:
                with open(affinity_file, 'r') as f:
                    affinity_data = json.load(f)
                
                # Extract key affinity metrics
                results['affinity_pred_value'] = affinity_data.get('affinity_pred_value', 0.0)
                results['binding_probability'] = affinity_data.get('affinity_probability_binary', 0.0)
                results['ic50_um'] = self._convert_affinity_to_ic50(affinity_data.get('affinity_pred_value', 0.0))
                
                logger.debug(f"Parsed affinity data: pred_value={results['affinity_pred_value']:.4f}, prob={results['binding_probability']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to parse affinity_data.json: {e}")
        
        # Parse confidence data (contains pLDDT and ipTM)
        confidence_file = os.path.join(result_dir, "confidence_data_model_0.json")
        if os.path.exists(confidence_file):
            try:
                with open(confidence_file, 'r') as f:
                    confidence_data = json.load(f)
                
                # Extract key confidence metrics
                results['plddt'] = confidence_data.get('complex_plddt', 0.0)
                results['iptm'] = confidence_data.get('iptm', 0.0)
                results['ptm'] = confidence_data.get('ptm', 0.0)
                results['confidence_score'] = confidence_data.get('confidence_score', 0.0)
                
                logger.debug(f"Parsed confidence data: pLDDT={results['plddt']:.4f}, ipTM={results['iptm']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to parse confidence_data_model_0.json: {e}")
        
        # Look for any additional files (backward compatibility)
        for filename in ['confidence.json', 'prediction_info.json']:
            file_path = os.path.join(result_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        additional_data = json.load(f)
                    results.update(additional_data)
                except Exception as e:
                    logger.warning(f"Failed to parse {filename}: {e}")
        
        logger.info(f"Parsed prediction results with {len(results)} metrics")
        return results
    
    def _convert_affinity_to_ic50(self, affinity_value: float) -> float:
        """Convert affinity prediction to IC50 (rough estimation)"""
        try:
            # This is a rough conversion - you may need to adjust based on your model
            # Higher affinity values typically correspond to lower IC50 values
            if affinity_value > 0:
                # Simple exponential conversion (adjust as needed)
                ic50_um = max(0.001, 1000.0 * (1.0 - affinity_value))
            else:
                ic50_um = 1000.0  # High IC50 for negative affinity
            return ic50_um
        except:
            return 0.0
    
    def _score_candidates(self, 
                         candidates: List[OptimizationCandidate],
                         reference_smiles: str) -> List[CompoundScore]:
        """Score all candidates"""
        scores = []
        
        for candidate in candidates:
            try:
                score = self.scoring_system.score_compound(
                    smiles=candidate.smiles,
                    boltz_results=candidate.prediction_results,
                    reference_smiles=reference_smiles
                )
                scores.append(score)
                candidate.scores = score
                
            except Exception as e:
                logger.error(f"Scoring failed for {candidate.compound_id}: {e}")
        
        logger.info(f"Scored {len(scores)} candidates")
        return scores
    
    def _prepare_top_candidates(self, 
                               ranked_scores: List[CompoundScore],
                               candidates: List[OptimizationCandidate]) -> List[Dict[str, Any]]:
        """Prepare top candidates for output"""
        top_candidates = []
        candidate_map = {c.smiles: c for c in candidates}
        
        for score in ranked_scores:
            candidate = candidate_map.get(score.smiles)
            if candidate:
                top_candidates.append({
                    'smiles': score.smiles,
                    'compound_id': candidate.compound_id,
                    'mmp_transformation': candidate.mmp_transformation,
                    'combined_score': score.combined_score,
                    'binding_affinity': score.binding_affinity,
                    'drug_likeness': score.drug_likeness,
                    'synthetic_accessibility': score.synthetic_accessibility,
                    'properties': score.properties
                })
        
        return top_candidates
    
    def _calculate_statistics(self, scores: List[CompoundScore]) -> Dict[str, Any]:
        """Calculate optimization statistics"""
        if not scores:
            return {}
        
        combined_scores = [s.combined_score for s in scores]
        
        return {
            'total_candidates': len(scores),
            'mean_score': sum(combined_scores) / len(combined_scores),
            'max_score': max(combined_scores),
            'min_score': min(combined_scores),
            'success_rate': len([s for s in scores if s.combined_score > 0.5]) / len(scores)
        }
    
    def _create_empty_result(self, compound_smiles: str, strategy: str, start_time: float) -> OptimizationResult:
        """Create empty result when no candidates are generated"""
        return OptimizationResult(
            original_compound=compound_smiles,
            strategy=strategy,
            candidates=[],
            scores=[],
            top_candidates=[],
            execution_time=time.time() - start_time,
            statistics={'total_candidates': 0},
            total_candidates=0,
            success_rate=0.0
        )
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def batch_optimize(self, 
                      compounds_file: str,
                      target_protein_yaml: str,
                      strategy: str = "scaffold_hopping",
                      max_candidates: int = 50,
                      output_dir: Optional[str] = None) -> Dict[str, OptimizationResult]:
        """
        Batch optimization of multiple compounds
        """
        results = {}
        
        # Setup output directory
        if not output_dir:
            output_dir = os.path.join(self.config.temp_dir, f"batch_optimization_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Read compounds file
        compounds = self._read_compounds_file(compounds_file)
        logger.info(f"Starting batch optimization for {len(compounds)} compounds")
        
        for i, (compound_id, smiles) in enumerate(compounds):
            logger.info(f"Optimizing compound {i+1}/{len(compounds)}: {compound_id}")
            
            compound_output_dir = os.path.join(output_dir, f"compound_{compound_id}")
            
            try:
                result = self.optimize_compound(
                    compound_smiles=smiles,
                    target_protein_yaml=target_protein_yaml,
                    strategy=strategy,
                    max_candidates=max_candidates,
                    output_dir=compound_output_dir
                )
                results[compound_id] = result
                logger.info(f"Compound {compound_id} optimization completed")
                
            except Exception as e:
                logger.error(f"Compound {compound_id} optimization failed: {e}")
                results[compound_id] = self._create_empty_result(smiles, strategy, time.time())
        
        logger.info(f"Batch optimization completed: {len(results)} compounds processed")
        return results
    
    def _read_compounds_file(self, compounds_file: str) -> List[Tuple[str, str]]:
        """Read compounds from file (CSV or text)"""
        compounds = []
        
        if compounds_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(compounds_file)
            for _, row in df.iterrows():
                compound_id = str(row.get('id', f'compound_{len(compounds)+1}'))
                smiles = str(row['smiles']).strip()
                compounds.append((compound_id, smiles))
        else:
            # Text file with SMILES
            with open(compounds_file, 'r') as f:
                for i, line in enumerate(f):
                    smiles = line.strip()
                    if smiles and not smiles.startswith('#'):
                        compounds.append((f'compound_{i+1}', smiles))
        
        return compounds
    
    def _candidate_to_dict(self, candidate):
        """Convert a candidate object to dictionary for JSON serialization"""
        result = {
            'compound_id': candidate.compound_id,
            'smiles': candidate.smiles,
            'combined_score': candidate.scores.combined_score if candidate.scores else 0.0,
            'mmp_transformation': candidate.mmp_transformation,
            'status': getattr(candidate, 'status', 'completed')
        }
        
        if candidate.scores:
            result.update({
                'binding_affinity': candidate.scores.binding_affinity,
                'drug_likeness': candidate.scores.drug_likeness,
                'synthetic_accessibility': candidate.scores.synthetic_accessibility,
                'novelty': candidate.scores.novelty,
                'stability': candidate.scores.stability,
                'plddt': candidate.scores.plddt,
                'iptm': candidate.scores.iptm,
                'binding_probability': candidate.scores.binding_probability,
                'ic50_um': candidate.scores.ic50_um,
                'molecular_weight': candidate.scores.molecular_weight,
                'logp': candidate.scores.logp,
                'lipinski_violations': candidate.scores.lipinski_violations,
                'qed_score': candidate.scores.qed_score
            })
        
        return result
    
    def _save_optimization_summary(self, result, output_dir: str):
        """Save optimization summary to JSON file"""
        import json
        
        summary = {
            'original_compound': result.original_compound,
            'strategy': result.strategy,
            'total_candidates': result.total_candidates,
            'successful_evaluations': len([c for c in result.candidates if c.get('status') == 'completed']),
            'success_rate': result.success_rate,
            'execution_time': result.execution_time,
            'best_candidate': result.top_candidates[0] if result.top_candidates else None,
            'top_candidates': result.top_candidates[:5],  # Top 5 candidates
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'strategy': result.strategy,
                'total_generated': result.total_candidates
            }
        }
        
        summary_file = os.path.join(output_dir, 'optimization_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimization summary saved to: {summary_file}")
