"""
RAG (Retrieval-Augmented Generation) Engine for BEP Generation

This module implements a RAG system that:
1. Extracts text from DOCX BEP documents
2. Creates embeddings and stores them in FAISS vector database
3. Retrieves relevant context for user queries
4. Generates contextual responses using Claude API
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
import logging
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BEPRAGEngine:
    """RAG Engine for BEP text generation"""

    def __init__(
        self,
        data_dir: str = 'data',
        vector_db_path: str = 'data/vector_db',
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialize RAG engine

        Args:
            data_dir: Directory containing training documents
            vector_db_path: Path to save/load FAISS vector database
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.data_dir = Path(data_dir)
        self.vector_db_path = Path(vector_db_path)
        self.txt_dir = self.data_dir / 'training_documents' / 'txt'

        # Set API key
        self.api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning(
                "No Anthropic API key provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None

        # Field-specific prompts
        self.field_contexts = {
            'executiveSummary': 'executive summary or project overview',
            'projectObjectives': 'project objectives and goals',
            'bimObjectives': 'BIM objectives and digital delivery goals',
            'projectScope': 'project scope and deliverables',
            'stakeholders': 'stakeholders, roles, and organizational structure',
            'rolesResponsibilities': 'roles, responsibilities, and task assignments',
            'deliveryTeam': 'delivery team structure and composition',
            'collaborationProcedures': 'collaboration procedures and workflows',
            'informationExchange': 'information exchange protocols and CDE',
            'cdeWorkflow': 'CDE workflow and information states',
            'modelRequirements': 'model requirements and LOD specifications',
            'dataStandards': 'data standards, schemas, and classification systems',
            'namingConventions': 'naming conventions and file organization',
            'qualityAssurance': 'quality assurance and validation procedures',
            'validationChecks': 'validation checks and quality control',
            'technologyStandards': 'technology standards and specifications',
            'softwarePlatforms': 'software platforms and tools',
            'coordinationProcess': 'coordination process and meetings',
            'clashDetection': 'clash detection and resolution procedures',
            'healthSafety': 'health and safety information requirements',
            'handoverRequirements': 'handover and closeout requirements',
            'asbuiltRequirements': 'as-built documentation requirements',
            'cobieRequirements': 'COBie and asset data requirements',
        }

    def initialize(self):
        """Initialize embeddings model and LLM"""
        logger.info("Initializing RAG engine...")

        # Initialize embeddings (using open-source model)
        logger.info("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize Claude LLM
        if self.api_key:
            logger.info("Initializing Claude LLM...")
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                anthropic_api_key=self.api_key,
                temperature=0.7,
                max_tokens=1024
            )
        else:
            logger.warning("Claude LLM not initialized (no API key)")

        logger.info("RAG engine initialized successfully")

    def load_or_create_vectorstore(self, force_rebuild: bool = False):
        """
        Load existing vector database or create new one

        Args:
            force_rebuild: Force rebuild even if database exists
        """
        # Check if vector database exists
        if self.vector_db_path.exists() and not force_rebuild:
            logger.info(f"Loading existing vector database from {self.vector_db_path}")
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.vector_db_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Vector database loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load vector database: {e}")
                logger.info("Will create new database...")

        # Create new vector database
        logger.info("Creating new vector database...")
        self._create_vectorstore()

    def _create_vectorstore(self):
        """Create vector database from text documents"""
        if not self.txt_dir.exists():
            raise FileNotFoundError(
                f"Training documents directory not found: {self.txt_dir}\n"
                "Please run: python scripts/extract_docx.py"
            )

        # Load documents
        logger.info(f"Loading documents from {self.txt_dir}")
        loader = DirectoryLoader(
            str(self.txt_dir),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        if not documents:
            raise ValueError("No documents found to create vector database")

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} text chunks")

        # Create vector database
        logger.info("Creating embeddings and building FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save vector database
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.vector_db_path))
        logger.info(f"Vector database saved to {self.vector_db_path}")

    def _get_field_prompt(self, field_type: str, partial_text: str = "") -> str:
        """
        Create field-specific prompt

        Args:
            field_type: Type of BEP field
            partial_text: User's partial input

        Returns:
            Formatted prompt
        """
        field_context = self.field_contexts.get(field_type, 'BEP content')

        prompt = f"""You are an expert in BIM Execution Plans (BEP) following ISO 19650 standards.

Generate professional, contextually appropriate text for the following section: {field_context}.

Requirements:
- Use formal, technical language appropriate for BEP documents
- Follow ISO 19650 terminology and standards
- Be specific and actionable
- Keep response concise (2-4 sentences)
- Continue naturally from any existing text provided

Context from similar BEP documents:
{{context}}

"""
        if partial_text:
            prompt += f"Existing text to continue from: {partial_text}\n\n"

        prompt += "Generate the next portion of text:\n"

        return prompt

    def generate_suggestion(
        self,
        field_type: str = 'default',
        partial_text: str = '',
        max_length: int = 300,
        k: int = 3
    ) -> Dict[str, str]:
        """
        Generate text suggestion using RAG

        Args:
            field_type: Type of BEP field
            partial_text: User's partial input
            max_length: Maximum characters to generate
            k: Number of relevant documents to retrieve

        Returns:
            Dictionary with 'text' and 'sources' keys
        """
        if not self.vectorstore:
            raise RuntimeError(
                "Vector database not loaded. "
                "Call load_or_create_vectorstore() first."
            )

        if not self.llm:
            raise RuntimeError(
                "Claude LLM not initialized. "
                "Please set ANTHROPIC_API_KEY environment variable."
            )

        # Create search query
        field_context = self.field_contexts.get(field_type, 'BEP content')
        if partial_text:
            query = f"{field_context}: {partial_text}"
        else:
            query = f"Examples of {field_context} in BEP documents"

        # Retrieve relevant documents
        logger.info(f"Searching for relevant context: {query[:100]}...")
        docs = self.vectorstore.similarity_search(query, k=k)

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt_template = self._get_field_prompt(field_type, partial_text)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context"]
        )

        # Generate response
        logger.info("Generating response with Claude...")
        formatted_prompt = prompt.format(context=context)

        response = self.llm.invoke(formatted_prompt)
        generated_text = response.content

        # Extract source information
        sources = [
            {
                'source': doc.metadata.get('source', 'Unknown'),
                'content': doc.page_content[:200] + '...'
            }
            for doc in docs
        ]

        logger.info(f"Generated {len(generated_text)} characters")

        return {
            'text': generated_text.strip(),
            'sources': sources,
            'retrieved_chunks': len(docs)
        }

    def get_status(self) -> Dict[str, any]:
        """Get RAG engine status"""
        return {
            'initialized': self.embeddings is not None,
            'vectorstore_loaded': self.vectorstore is not None,
            'llm_available': self.llm is not None,
            'api_key_configured': bool(self.api_key),
            'vector_db_path': str(self.vector_db_path),
            'documents_path': str(self.txt_dir)
        }


# Global instance
_rag_engine = None


def get_rag_engine() -> BEPRAGEngine:
    """Get or create global RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        script_dir = Path(__file__).parent
        data_dir = script_dir / 'data'

        _rag_engine = BEPRAGEngine(data_dir=str(data_dir))
        _rag_engine.initialize()

        try:
            _rag_engine.load_or_create_vectorstore()
        except Exception as e:
            logger.error(f"Failed to load/create vector database: {e}")

    return _rag_engine


if __name__ == "__main__":
    # Test the RAG engine
    logger.info("Testing RAG Engine...")

    engine = BEPRAGEngine()
    engine.initialize()
    engine.load_or_create_vectorstore(force_rebuild=True)

    # Test generation
    result = engine.generate_suggestion(
        field_type='executiveSummary',
        partial_text='This BEP establishes',
        k=3
    )

    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(result['text'])
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['source']}")
        print(f"   {source['content'][:150]}...")
