import os
import sys
import asyncio
import json
import logging
from pathlib import Path

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops if running in some environments
nest_asyncio.apply()

# Load env before imports that might need it
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE)

# Langchain / Ragas imports
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.run_config import RunConfig

# App imports
from pipeline.session import session_store
from pipeline.ingestion import ingest_pdf
from pipeline.retrieval import retrieve
from pipeline.generation import generate_streaming
from pipeline.memory import ChatMemory
from config import get_settings
from groq import AsyncGroq, Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    settings = get_settings()
    pdf_path = str(_PROJECT_ROOT / "7181-attention-is-all-you-need-2.pdf")

    # Testing overrides to aggressively save tokens and avoid Groq rate limits
    settings.chunk_size = 350
    settings.top_k_rerank = 3
    test_max_tokens = 250

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print("1. Initialising models and clients...")
    critic_llm = ChatGroq(
        model=settings.groq_model_primary,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=512,
        n=1
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    groq_sync  = Groq(api_key=settings.groq_api_key)
    groq_async = AsyncGroq(api_key=settings.groq_api_key)

    golden_testset = [
        {
            "question": "What is the primary advantage of the Transformer architecture over RNNs or CNNs?",
            "ground_truth": "The Transformer relies entirely on an attention mechanism to draw global dependencies between input and output, allowing for significantly more parallelization and reducing training times compared to recurrent layers."
        },
        {
            "question": "What are the components of the Multi-Head Attention mechanism?",
            "ground_truth": "Multi-head attention consists of several parallel attention layers or 'heads'. It takes the Query, Key, and Value matrices, linearly projects them multiple times, applies Scaled Dot-Product Attention in parallel, concatenates the outputs, and linearly projects them again."
        },
        {
            "question": "How does the model handle the order of the sequence without using recurrence?",
            "ground_truth": "The model uses 'Positional Encoding' added to the input embeddings at the bottoms of the encoder and decoder stacks. These encodings use sine and cosine functions of different frequencies to inject information about the relative or absolute position of the tokens."
        },
        {
            "question": "What optimizer was used for training the Transformer model?",
            "ground_truth": "The model was trained using the Adam optimizer with parameters beta1 = 0.9, beta2 = 0.98, and epsilon = 10^-9, varying the learning rate over the course of training with a warmup stage."
        },
        {
            "question": "Which dataset was used for the English-to-German translation task?",
            "ground_truth": "The WMT 2014 English-to-German dataset consisting of about 4.5 million sentence pairs was used."
        },
        {
            "question": "What BLEU score did the big Transformer model achieve on English-to-German translation?",
            "ground_truth": "The big Transformer model achieved 28.4 BLEU on the WMT 2014 English-to-German translation task."
        },
        {
            "question": "What regularization techniques were used during training?",
            "ground_truth": "Three regularization techniques were used: residual dropout applied to the output of each sub-layer, dropout applied to the sums of embeddings and positional encodings, and label smoothing with a value of 0.1."
        },
        {
            "question": "What is the formula for Scaled Dot-Product Attention?",
            "ground_truth": "The output is computed as softmax(QK^T / sqrt(d_k))V."
        },
        {
            "question": "Why does the model employ a scaling factor of 1/sqrt(d_k) in the attention mechanism?",
            "ground_truth": "To counteract the effect of large dot products pushing the softmax function into regions with extremely small gradients."
        },
        {
            "question": "What is the dimension of the keys and queries (d_k) and the values (d_v) in the multi-head attention layers?",
            "ground_truth": "In the paper, d_k = d_v = d_model / h = 64, where h is the number of heads (8) and d_model is 512."
        },
        {
            "question": "How is the feed-forward network in each encoder and decoder layer structured?",
            "ground_truth": "It consists of two linear transformations with a ReLU activation in between: FFN(x) = max(0, xW1 + b1)W2 + b2."
        },
        {
            "question": "What is the dimensionality of the inner layer (d_ff) in the position-wise feed-forward networks?",
            "ground_truth": "The inner layer has a dimensionality of d_ff = 2048."
        },
        {
            "question": "How many layers do the encoder and decoder stacks have?",
            "ground_truth": "Both the encoder and decoder are composed of a stack of N = 6 identical layers."
        },
        {
            "question": "What BLEU score did the base Transformer model achieve on English-to-German translation?",
            "ground_truth": "The base Transformer model achieved a BLEU score of 27.3 on the WMT 2014 English-to-German translation task."
        },
        {
            "question": "What hardware was used for training the models?",
            "ground_truth": "The models were trained on one machine with 8 NVIDIA P100 GPUs."
        }
    ]

    print("2. Ingesting PDF...")
    session_id = "eval_session"
    # mirror exactly how main.py creates sessions
    session = session_store.create(session_id)
    with open(pdf_path, "rb") as f:
        await ingest_pdf(f.read(), session, groq_sync)

    print("3. Running questions through pipeline...\n")
    answers, contexts = [], []

    for row in golden_testset:
        question = row["question"]
        memory = ChatMemory()
        retrieval_result = retrieve(question, session)

        response_text = ""
        stream = generate_streaming(
            query=question,
            retrieval=retrieval_result,
            memory=memory,
            groq_client=groq_async,
            doc_title=session.doc_title,
            max_tokens=test_max_tokens,
        )
        
        async for chunk in stream:
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if "token" in data:
                        response_text += data["token"]
                except Exception as e:
                    logger.debug(f"SSE parse error: {e} | chunk: {chunk}")

        print(f"  Q: {question[:80]}")
        print(f"  Chunks: {len(retrieval_result.chunks)} | Confident: {retrieval_result.confident}")
        print(f"  A: {response_text[:120]}...\n")
        answers.append(response_text)
        contexts.append([c.text for c in retrieval_result.chunks])

        # Delay to avoid hitting Groq/Cohere rate limits (e.g. 30 RPM)
        print("  [Waiting 12 seconds to respect API quotas...]")
        await asyncio.sleep(12)

    print("4. Filtering and evaluating...")
    evaluable = [i for i, c in enumerate(contexts) if c]
    skipped = len(contexts) - len(evaluable)
    if skipped:
        print(f"  Skipping {skipped} confidence-gate samples")

    # Groq Free Tier allows 30 Requests Per Minute (RPM). 
    # Each question evaluates 4 metrics, triggering ~5 to 8 LLM calls.
    # We evaluate in batches of 3 questions (~15-24 calls) and sleep for 65 seconds to respect the quota.
    batch_size = 3
    all_results_df = pd.DataFrame()
    run_config = RunConfig(max_workers=2, max_retries=5)
    
    for i in range(0, len(evaluable), batch_size):
        batch_indices = evaluable[i:i + batch_size]
        
        batch_dataset = Dataset.from_dict({
            "question":     [golden_testset[j]["question"] for j in batch_indices],
            "answer":       [answers[j] for j in batch_indices],
            "contexts":     [contexts[j] for j in batch_indices],
            "ground_truth": [golden_testset[j]["ground_truth"] for j in batch_indices],
        })
        
        print(f"\n  [Evaluating Batch {i // batch_size + 1} of {(len(evaluable) + batch_size - 1) // batch_size} ({len(batch_indices)} questions)...]")
        try:
            batch_result = evaluate(
                batch_dataset,
                metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
                llm=critic_llm,
                embeddings=embeddings,
                run_config=run_config,
            )
            # Combine the results into a single DataFrame
            all_results_df = pd.concat([all_results_df, batch_result.to_pandas()], ignore_index=True)
            
        except Exception as e:
            print(f"Evaluation failed on Batch {i // batch_size + 1}: {e}")
            raise
            
        if i + batch_size < len(evaluable):
            print("  [Batch complete. Sleeping 65 seconds to reset Groq 30 RPM quota...]")
            await asyncio.sleep(65)

    try:
        aggregate = {
            "faithfulness":      round(all_results_df["faithfulness"].mean(), 4) if "faithfulness" in all_results_df else 0.0,
            "answer_relevancy":  round(all_results_df["answer_relevancy"].mean(), 4) if "answer_relevancy" in all_results_df else 0.0,
            "context_precision": round(all_results_df["context_precision"].mean(), 4) if "context_precision" in all_results_df else 0.0,
            "context_recall":    round(all_results_df["context_recall"].mean(), 4) if "context_recall" in all_results_df else 0.0,
            "samples_evaluated": len(all_results_df),
            "model":             settings.groq_model_primary,
            "critic_model":      settings.groq_model_primary,
            "embed_model":       settings.embed_model,
        }

        print("\n=== RAGAS Results ===")
        for k, v in aggregate.items():
            print(f"  {k}: {v}")

        # Save outputs
        all_results_df.to_csv(_PROJECT_ROOT / "ragas_results.csv", index=False)
        json_path = str(_PROJECT_ROOT / "ragas_scores.json")
        with open(json_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"\nSaved: ragas_results.csv + {json_path}")

    except Exception as e:
        print(f"Aggregation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
