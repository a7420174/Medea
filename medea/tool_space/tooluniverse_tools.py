"""
Medea Tools for ToolUniverse

This module wraps Medea's existing biological research tools to be compatible
with ToolUniverse framework, following the BaseTool pattern.

Each tool class wraps an existing Medea function without modifying the original.
"""

from tooluniverse.tool_registry import register_tool
from tooluniverse.base_tool import BaseTool
from typing import Dict, Any
import json
from pathlib import Path


# Tools use lazy imports (inside run() method) to avoid circular dependencies

# ============================================================================
# Disease & Target Analysis Tools
# ============================================================================

@register_tool('load_disease_targets')
class LoadDiseaseTargetsTool(BaseTool):
    """Load disease-associated therapeutic targets from OpenTargets."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool to retrieve disease targets."""
        from .action_functions import load_disease_targets
        
        disease_name = arguments.get('disease_name')
        use_api = arguments.get('use_api', True)
        attributes = arguments.get('attributes', ["otGeneticsPortal", "chembl"])
        
        try:
            targets = load_disease_targets(
                disease_name=disease_name,
                use_api=use_api,
                attributes=attributes
            )
            return {
                "success": True,
                "targets": list(targets),
                "count": len(targets),
                "disease": disease_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "disease": disease_name
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'disease_name' not in kwargs or not kwargs['disease_name']:
            raise ValueError("disease_name is required")


@register_tool('load_pinnacle_ppi')
class LoadPinnaclePPITool(BaseTool):
    """Load cell-type-specific protein embeddings from PINNACLE."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool to retrieve PINNACLE embeddings."""
        from .action_functions import load_pinnacle_ppi
        
        cell_type = arguments.get('cell_type')
        
        try:
            embeddings = load_pinnacle_ppi(cell_type=cell_type)
            return {
                "success": True,
                "cell_type": cell_type,
                "num_genes": len(embeddings),
                "genes": list(embeddings.keys())[:100],  # First 100 genes
                "embedding_shape": str(list(embeddings.values())[0].shape) if embeddings else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "cell_type": cell_type
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'cell_type' not in kwargs or not kwargs['cell_type']:
            raise ValueError("cell_type is required")


# ============================================================================
# HumanBase Network Analysis Tools
# ============================================================================

@register_tool('humanbase_analyze_tissue_coexpression')
class HumanBaseCoexpressionTool(BaseTool):
    """Analyze tissue-specific gene co-expression using HumanBase."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tissue co-expression analysis."""
        from .humanbase import humanbase_analyze_tissue_coexpression
        
        genes = arguments.get('genes', [])
        tissue = arguments.get('tissue')
        max_interactions = arguments.get('max_interactions', 10)
        
        try:
            # Returns tuple: (summary, confidence, top_interactions, biological_processes)
            summary, confidence, top_interactions, biological_processes = humanbase_analyze_tissue_coexpression(
                genes=genes,
                tissue=tissue,
                max_interactions=max_interactions
            )
            return {
                "success": True,
                "tissue": tissue,
                "num_genes": len(genes),
                "summary": summary,
                "confidence": confidence,
                "top_interactions": top_interactions,
                "biological_processes": biological_processes
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'genes' not in kwargs or not kwargs['genes']:
            raise ValueError("genes list is required")
        if 'tissue' not in kwargs or not kwargs['tissue']:
            raise ValueError("tissue is required")


@register_tool('humanbase_analyze_tissue_protein_interactions')
class HumanBaseProteinInteractionsTool(BaseTool):
    """Analyze tissue-specific protein-protein interactions using HumanBase."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protein interaction analysis."""
        from .humanbase import humanbase_analyze_tissue_protein_interactions
        
        genes = arguments.get('genes', [])
        tissue = arguments.get('tissue')
        max_interactions = arguments.get('max_interactions', 10)
        
        try:
            # Returns tuple: (summary, confidence, top_interactions, biological_processes)
            summary, confidence, top_interactions, biological_processes = humanbase_analyze_tissue_protein_interactions(
                genes=genes,
                tissue=tissue,
                max_interactions=max_interactions
            )
            return {
                "success": True,
                "tissue": tissue,
                "num_genes": len(genes),
                "summary": summary,
                "confidence": confidence,
                "top_interactions": top_interactions,
                "biological_processes": biological_processes
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'genes' not in kwargs or not kwargs['genes']:
            raise ValueError("genes list is required")
        if 'tissue' not in kwargs or not kwargs['tissue']:
            raise ValueError("tissue is required")


# ============================================================================
# Pathway & Functional Analysis Tools
# ============================================================================

@register_tool('analyze_pathway_interaction')
class AnalyzePathwayTool(BaseTool):
    """Analyze genetic pathway interactions between two genes."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pathway interaction analysis."""
        from .enrichr import analyze_pathway_interaction
        
        gene1 = arguments.get('gene1')
        gene2 = arguments.get('gene2')
        
        try:
            # Returns tuple: (summary, confidence, mechanisms, pathways)
            summary, confidence, mechanisms, pathways = analyze_pathway_interaction(
                gene1=gene1,
                gene2=gene2
            )
            return {
                "success": True,
                "gene1": gene1,
                "gene2": gene2,
                "summary": summary,
                "confidence": confidence,
                "mechanisms": mechanisms,
                "pathways": pathways
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'gene1' not in kwargs or not kwargs['gene1']:
            raise ValueError("gene1 is required")
        if 'gene2' not in kwargs or not kwargs['gene2']:
            raise ValueError("gene2 is required")


@register_tool('analyze_comprehensive_interaction')
class AnalyzeComprehensiveTool(BaseTool):
    """Comprehensive interaction analysis across all biological databases."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive analysis across pathway, reactome, hallmark, function, and process databases."""
        from .enrichr import analyze_comprehensive_interaction
        
        gene1 = arguments.get('gene1')
        gene2 = arguments.get('gene2')
        
        try:
            # Returns dict with keys: 'pathway', 'reactome', 'hallmark', 'function', 'process'
            # Each value is a tuple: (summary, confidence, mechanisms, pathways)
            results = analyze_comprehensive_interaction(gene1=gene1, gene2=gene2)
            
            # Convert tuples to dicts for better JSON serialization
            formatted_results = {}
            for db_name, (summary, confidence, mechanisms, pathways) in results.items():
                formatted_results[db_name] = {
                    "summary": summary,
                    "confidence": confidence,
                    "mechanisms": mechanisms,
                    "pathways": pathways
                }
            
            return {
                "success": True,
                "gene1": gene1,
                "gene2": gene2,
                "analyses": formatted_results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'gene1' not in kwargs or not kwargs['gene1']:
            raise ValueError("gene1 is required")
        if 'gene2' not in kwargs or not kwargs['gene2']:
            raise ValueError("gene2 is required")


# ============================================================================
# Machine Learning Prediction Tools
# ============================================================================

@register_tool('compass_predict')
class CompassPredictTool(BaseTool):
    """COMPASS model for ICI response prediction and immune concept extraction."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute COMPASS prediction on patient transcriptomic data."""
        from .compass import compass_predict
        
        df_tpm = arguments.get('df_tpm')
        root_path = arguments.get('root_path')
        ckp_path = arguments.get('ckp_path', 'pft_leave_IMVigor210.pt')
        device = arguments.get('device', 'cuda')
        threshold = arguments.get('threshold', 0.5)
        batch_size = arguments.get('batch_size', 128)
        
        try:
            # Returns tuple: (responder: bool, cell_concepts: list[tuple[str, float]])
            responder, cell_concepts = compass_predict(
                df_tpm=df_tpm,
                root_path=root_path,
                ckp_path=ckp_path,
                device=device,
                threshold=threshold,
                batch_size=batch_size
            )
            return {
                "success": True,
                "responder": responder,
                "cell_concepts": cell_concepts,
                "num_concepts": len(cell_concepts)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'df_tpm' not in kwargs or kwargs['df_tpm'] is None:
            raise ValueError("df_tpm (patient transcriptomic DataFrame) is required")


@register_tool('compute_depmap_correlations')
class DepMapCorrelationsTool(BaseTool):
    """Compute gene dependency correlations from DepMap database."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DepMap correlation analysis for gene pair."""
        from .depmap import compute_depmap24q2_gene_correlations
        
        gene_a = arguments.get('gene_a')
        gene_b = arguments.get('gene_b')
        
        try:
            result = compute_depmap24q2_gene_correlations(gene_a=gene_a, gene_b=gene_b)
            return {
                "success": True,
                "gene_a": gene_a,
                "gene_b": gene_b,
                "correlation": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        if 'gene_a' not in kwargs or not kwargs['gene_a']:
            raise ValueError("gene_a is required")
        if 'gene_b' not in kwargs or not kwargs['gene_b']:
            raise ValueError("gene_b is required")


# ============================================================================
# Yeast-Human Ortholog Mapping Tools
# ============================================================================

@register_tool('find_human_orthologs_for_yeast_gene')
class FindHumanOrthologsForYeastGeneTool(BaseTool):
    """Find human orthologs for a yeast gene using SGD (direct) + PomBase (triangulated via S. pombe)."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Look up human orthologs for a S. cerevisiae gene."""
        from .yeast_human_orthologs import find_human_orthologs_for_yeast_gene
        
        yeast_gene = arguments.get('yeast_gene')
        include_pombase = arguments.get('include_pombase', True)
        
        try:
            return find_human_orthologs_for_yeast_gene(
                yeast_gene=yeast_gene,
                include_pombase=include_pombase
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "yeast_gene": yeast_gene
            }
    
    def validate_input(self, **kwargs) -> None:
        if 'yeast_gene' not in kwargs or not kwargs['yeast_gene']:
            raise ValueError("yeast_gene is required (standard name like 'RAD51' or systematic name like 'YER095W')")


@register_tool('find_yeast_orthologs_for_human_gene')
class FindYeastOrthologsForHumanGeneTool(BaseTool):
    """Find yeast (S. cerevisiae) orthologs for a human gene using PomBase triangulation + SGD validation."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Look up yeast orthologs for a human gene."""
        from .yeast_human_orthologs import find_yeast_orthologs_for_human_gene
        
        human_gene = arguments.get('human_gene')
        include_pombase = arguments.get('include_pombase', True)
        
        try:
            return find_yeast_orthologs_for_human_gene(
                human_gene=human_gene,
                include_pombase=include_pombase
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "human_gene": human_gene
            }
    
    def validate_input(self, **kwargs) -> None:
        if 'human_gene' not in kwargs or not kwargs['human_gene']:
            raise ValueError("human_gene is required (e.g. 'TP53', 'BRCA1')")


@register_tool('find_yeast_human_complementation')
class FindYeastHumanComplementationTool(BaseTool):
    """Find experimentally validated functional complementation pairs between yeast and human genes."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Look up functional complementation data from SGD."""
        from .yeast_human_orthologs import find_yeast_human_complementation
        
        yeast_gene = arguments.get('yeast_gene')
        
        try:
            return find_yeast_human_complementation(yeast_gene=yeast_gene)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "yeast_gene": yeast_gene
            }
    
    def validate_input(self, **kwargs) -> None:
        if 'yeast_gene' not in kwargs or not kwargs['yeast_gene']:
            raise ValueError("yeast_gene is required")


@register_tool('get_yeast_gene_info')
class GetYeastGeneInfoTool(BaseTool):
    """Get comprehensive information about a yeast gene from SGD (description, type, aliases, GO annotations)."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve yeast gene information from SGD."""
        from .yeast_human_orthologs import get_yeast_gene_info
        
        yeast_gene = arguments.get('yeast_gene')
        
        try:
            return get_yeast_gene_info(yeast_gene=yeast_gene)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "yeast_gene": yeast_gene
            }
    
    def validate_input(self, **kwargs) -> None:
        if 'yeast_gene' not in kwargs or not kwargs['yeast_gene']:
            raise ValueError("yeast_gene is required (standard name like 'RAD51' or systematic name like 'YER095W')")


@register_tool('batch_yeast_to_human_mapping')
class BatchYeastToHumanMappingTool(BaseTool):
    """Batch-map multiple yeast genes to their human orthologs using SGD + PomBase."""
    
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch ortholog mapping."""
        from .yeast_human_orthologs import batch_yeast_to_human_mapping
        
        yeast_genes = arguments.get('yeast_genes', [])
        include_pombase = arguments.get('include_pombase', True)
        
        try:
            return batch_yeast_to_human_mapping(
                yeast_genes=yeast_genes,
                include_pombase=include_pombase
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_input(self, **kwargs) -> None:
        if 'yeast_genes' not in kwargs or not kwargs['yeast_genes']:
            raise ValueError("yeast_genes list is required (e.g. ['RAD51', 'CDC28', 'TUB1'])")


# ============================================================================
# Registry Helper
# ============================================================================

def get_registered_tools():
    """Get all registered Medea tools for ToolUniverse."""
    from tooluniverse.tool_registry import get_tool_registry
    return get_tool_registry()


def list_medea_tools():
    """List all Medea tool names registered with ToolUniverse."""
    tools = get_registered_tools()
    medea_tools = [name for name in tools.keys() if any(
        name.startswith(prefix) for prefix in [
            'load_', 'humanbase_', 'analyze_', 'compass_', 
            'compute_', 'find_human_orthologs', 'find_yeast_orthologs',
            'find_yeast_human', 'get_yeast_gene', 'batch_yeast'
        ]
    )]
    return medea_tools

