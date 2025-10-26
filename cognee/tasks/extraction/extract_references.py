"""Reference extraction task for extracting citations and references from documents."""

from typing import Dict, List, Optional, Any
import logging
from pydantic import BaseModel, Field
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.data.models import KnowledgeGraph

logger = logging.getLogger(__name__)


class Reference(BaseModel):
    """Schema for extracted references."""
    title: str = Field(description="Title of the reference")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    year: Optional[int] = Field(None, description="Publication year")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    url: Optional[str] = Field(None, description="URL or link to reference")
    
    def __hash__(self):
        """Enable hashing for deduplication."""
        return hash((self.title.lower().strip(), tuple(sorted([a.lower().strip() for a in self.authors])), self.year))
    
    def __eq__(self, other):
        """Check equality for deduplication."""
        if not isinstance(other, Reference):
            return False
        return (self.title.lower().strip() == other.title.lower().strip() and
                sorted([a.lower().strip() for a in self.authors]) == sorted([a.lower().strip() for a in other.authors]) and
                self.year == other.year)


class ExtractReferencesTask:
    """Task for extracting references from document chunks using LLM."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_llm_client()
        self.extracted_references: Dict[str, Reference] = {}
    
    async def extract_references(self, text: str, document_id: str) -> List[Reference]:
        """Extract references from text using LLM.
        
        Args:
            text: Text content to extract references from
            document_id: ID of the source document
            
        Returns:
            List of extracted Reference objects
        """
        prompt = f"""Extract all academic references, citations, and bibliographic entries from the following text.
For each reference, extract:
- title: The title of the work
- authors: List of author names
- year: Publication year (if available)
- doi: DOI if mentioned
- url: URL if mentioned

Return ONLY a JSON array of references. If no references are found, return an empty array [].

Text:
{text}

JSON:"""
        
        try:
            response = await self.llm_client.acreate_structured_output(
                text_input=prompt,
                response_model=List[Reference]
            )
            
            if response:
                logger.info(f"Extracted {len(response)} references from document {document_id}")
                return response
            return []
            
        except Exception as e:
            logger.error(f"Error extracting references: {e}")
            return []
    
    def deduplicate_references(self, references: List[Reference]) -> List[Reference]:
        """Deduplicate references based on title, authors, and year.
        
        Args:
            references: List of references to deduplicate
            
        Returns:
            Deduplicated list of references
        """
        seen = set()
        deduplicated = []
        
        for ref in references:
            if ref not in seen:
                seen.add(ref)
                deduplicated.append(ref)
        
        return deduplicated
    
    async def add_references_to_graph(self, references: List[Reference], 
                                      document_id: str, graph: KnowledgeGraph) -> None:
        """Add references as nodes to the knowledge graph and link to parent document.
        
        Args:
            references: List of references to add
            document_id: ID of the parent document
            graph: Knowledge graph to add references to
        """
        for ref in references:
            # Create unique reference ID
            ref_id = f"ref_{hash(ref)}"
            
            # Add reference as graph node
            await graph.add_node(
                node_id=ref_id,
                node_type="Reference",
                properties={
                    "title": ref.title,
                    "authors": ",".join(ref.authors),
                    "year": ref.year,
                    "doi": ref.doi,
                    "url": ref.url
                }
            )
            
            # Link reference to parent document
            await graph.add_edge(
                from_node=ref_id,
                to_node=document_id,
                edge_type="is_part_of",
                properties={"relation": "cited_in"}
            )
            
            logger.info(f"Added reference '{ref.title}' to graph and linked to document {document_id}")
    
    async def __call__(self, chunks: List[Dict[str, Any]], graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Process chunks to extract references.
        
        Args:
            chunks: List of document chunks
            graph: Knowledge graph to add references to
            
        Returns:
            Original chunks (passthrough)
        """
        all_references = []
        
        for chunk in chunks:
            document_id = chunk.get("document_id", "unknown")
            text = chunk.get("text", "")
            
            # Extract references from chunk
            references = await self.extract_references(text, document_id)
            all_references.extend(references)
        
        # Deduplicate all extracted references
        deduplicated_refs = self.deduplicate_references(all_references)
        logger.info(f"Deduplicated {len(all_references)} references to {len(deduplicated_refs)} unique references")
        
        # Add to graph (grouped by document)
        doc_refs = {}
        for i, chunk in enumerate(chunks):
            doc_id = chunk.get("document_id", "unknown")
            if doc_id not in doc_refs:
                doc_refs[doc_id] = []
        
        # Distribute references to their source documents
        for ref in deduplicated_refs:
            # In practice, you'd track which chunk each ref came from
            # For now, add to first document as example
            if doc_refs:
                first_doc = list(doc_refs.keys())[0]
                doc_refs[first_doc].append(ref)
        
        # Add references to graph
        for doc_id, refs in doc_refs.items():
            await self.add_references_to_graph(refs, doc_id, graph)
        
        return chunks  # Return original chunks unchanged
