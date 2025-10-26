"""Example: Using Reference Extraction in Custom Pipeline

This example demonstrates how to use the reference extraction task
to extract citations and references from documents and add them to
the knowledge graph.

Issue: #633
"""

import asyncio
from cognee import cognify
from cognee.tasks.extraction.extract_references import ExtractReferencesTask
from cognee.modules.pipelines import Pipeline


async def main():
    """
    Example of using custom pipeline with reference extraction.
    
    This keeps the default cognee pipeline unchanged while allowing
    users to add reference extraction as an optional step.
    """
    
    # Example document with references
    sample_text = """
    Recent advances in machine learning have shown promising results.
    According to Smith et al. (2023), transformer models continue to
    improve in various NLP tasks. The original attention mechanism was
    introduced by Vaswani et al. in 2017 in their paper "Attention is
    All You Need" (https://arxiv.org/abs/1706.03762).
    
    References:
    1. Smith, J., Doe, A., & Johnson, B. (2023). Advanced Transformers
       for Natural Language Processing. Journal of ML Research, 45(3), 123-145.
       DOI: 10.1234/jmlr.2023.5678
    
    2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is
       All You Need. Advances in Neural Information Processing Systems.
    """
    
    # Create custom task list with reference extraction
    # This is added AFTER document chunking in the pipeline
    custom_tasks = [
        # ... standard cognee tasks (document processing, chunking, etc.)
        # Then add reference extraction:
        ExtractReferencesTask(),
        # ... continue with other tasks (summarization, etc.)
    ]
    
    # Option 1: Use custom pipeline (recommended approach)
    # custom_pipeline = Pipeline(tasks=custom_tasks)
    # await cognify(sample_text, pipeline=custom_pipeline)
    
    # Option 2: For this example, we'll demonstrate the task directly
    print("Extracting references from sample text...")
    
    # Initialize the reference extraction task
    ref_extractor = ExtractReferencesTask()
    
    # Simulate document chunks (normally these come from the pipeline)
    chunks = [
        {
            "document_id": "doc_001",
            "text": sample_text,
            "chunk_index": 0
        }
    ]
    
    # Extract references (requires graph instance in real usage)
    # In actual implementation, this would be called within the pipeline
    # with access to the knowledge graph
    references = await ref_extractor.extract_references(
        text=sample_text,
        document_id="doc_001"
    )
    
    print(f"\nExtracted {len(references)} references:")
    for i, ref in enumerate(references, 1):
        print(f"\n{i}. {ref.title}")
        print(f"   Authors: {', '.join(ref.authors) if ref.authors else 'N/A'}")
        print(f"   Year: {ref.year or 'N/A'}")
        if ref.doi:
            print(f"   DOI: {ref.doi}")
        if ref.url:
            print(f"   URL: {ref.url}")
    
    # Demonstrate deduplication
    print("\n" + "="*50)
    print("Testing deduplication...")
    duplicated_refs = references + references[:1]  # Add a duplicate
    deduplicated = ref_extractor.deduplicate_references(duplicated_refs)
    print(f"Before deduplication: {len(duplicated_refs)} references")
    print(f"After deduplication: {len(deduplicated)} references")
    
    print("\n" + "="*50)
    print("\nTo use in production:")
    print("1. Create a custom task list including ExtractReferencesTask()")
    print("2. Pass it to cognify via the pipeline parameter")
    print("3. References will be added to the knowledge graph")
    print("4. Query the graph to retrieve documents with their citations")
    print("\nExample:")
    print("  custom_pipeline = Pipeline(tasks=[...standard_tasks..., ExtractReferencesTask(), ...])")
    print("  await cognify(documents, pipeline=custom_pipeline)")


if __name__ == "__main__":
    asyncio.run(main())
