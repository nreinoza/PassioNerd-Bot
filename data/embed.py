import pandas as pd
import numpy as np
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

async def transform_text_batch(
    client: AsyncOpenAI,
    texts: List[str],
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> List[Optional[str]]:
    """Transform a batch of texts concurrently."""

    async def transform_single(text: str, idx: int) -> tuple[int, Optional[str]]:
        async with semaphore:
            try:
                full_prompt = prompt.replace("{text}", str(text))
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    reasoning_effort="minimal",
                    verbosity="low",
                )
                return idx, response.choices[0].message.content
            except Exception as e:
                print(f"\nError at row {idx}: {str(e)[:100]}")
                return idx, None

    tasks = [transform_single(text, idx) for idx, text in enumerate(texts)]

    results = await tqdm.gather(*tasks, desc="Transforming texts", unit="text")

    sorted_results = sorted(results, key=lambda x: x[0])
    return [result[1] for result in sorted_results]


async def transform_text(
    df: pd.DataFrame,
    text_column: str,
    prompt: str,
    api_key: str,
    output_column: str = "transformed_text",
    model: str = "gpt-5-nano",
    max_concurrent: int = 50,
) -> pd.DataFrame:
    """Transform text using OpenAI API with concurrent requests."""

    print(f"\nSTEP 1: Transforming text in column '{text_column}'")
    print(f"Model: {model} | Rows: {len(df)} | Concurrency: {max_concurrent}")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    texts = df[text_column].tolist()
    transformed_texts = await transform_text_batch(
        client, texts, prompt, model, semaphore
    )

    df[output_column] = transformed_texts

    success_count = sum(1 for t in transformed_texts if t is not None)
    print(f"Transformation complete: {success_count}/{len(df)} rows")

    return df


async def generate_embeddings_batch(
    client: AsyncOpenAI,
    texts: List[str],
    embedding_model: str,
    batch_size: int,
    semaphore: asyncio.Semaphore,
) -> List[Optional[List[float]]]:
    """Generate embeddings for texts concurrently in batches."""

    async def process_batch(
        batch_texts: List[str], start_idx: int
    ) -> tuple[int, List[Optional[List[float]]]]:
        async with semaphore:
            try:
                clean_texts = [
                    str(text) if text is not None else "" for text in batch_texts
                ]
                response = await client.embeddings.create(
                    model=embedding_model, input=clean_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                return start_idx, batch_embeddings
            except Exception as e:
                print(f"\nError in batch at index {start_idx}: {str(e)[:100]}")
                return start_idx, [None] * len(batch_texts)

    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tasks.append(process_batch(batch, i))

    results = await tqdm.gather(*tasks, desc="Generating embeddings", unit="batch")

    sorted_results = sorted(results, key=lambda x: x[0])
    all_embeddings = []
    for _, embeddings in sorted_results:
        all_embeddings.extend(embeddings)

    return all_embeddings


async def generate_embeddings(
    df: pd.DataFrame,
    text_column: str,
    api_key: str,
    output_column: str = "embedding",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_concurrent: int = 10,
) -> pd.DataFrame:
    """Generate embeddings using OpenAI API with concurrent batch requests."""

    print(f"\nSTEP 2: Generating embeddings for column '{text_column}'")
    print(
        f"Model: {embedding_model} | Batch size: {batch_size} | Concurrency: {max_concurrent}"
    )

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    texts = df[text_column].tolist()
    embeddings = await generate_embeddings_batch(
        client, texts, embedding_model, batch_size, semaphore
    )

    df[output_column] = embeddings

    success_count = sum(1 for e in embeddings if e is not None)
    print(f"Embedding generation complete: {success_count}/{len(df)} rows")

    return df

async def process_csv(
    input_csv: str,
    output_csv: str,
    text_columns: List[str],
    transformation_prompt: str,
    api_key: str,
    model: str = "gpt-5-nano",
    embedding_model: str = "text-embedding-3-small",
    dimensionality_methods: Optional[List[Dict]] = None,
    random_sample: Optional[int] = None,
    concatenated_column: str = "concatenated_text",
    max_concurrent_transforms: int = 50,
    max_concurrent_embeddings: int = 10,
    embedding_batch_size: int = 100,
):
    """Main async pipeline to process CSV file."""

    print(f"\nTEXT PROCESSING PIPELINE")
    print(f"Input: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} rows")

    if random_sample is not None and random_sample < len(df):
        print(f"Randomly sampling {random_sample:,} rows")
        df = df.sample(n=random_sample, random_state=42).reset_index(drop=True)
        print(f"Sample size: {len(df):,} rows")

    if isinstance(text_columns, str):
        text_columns = [text_columns]

    print(f"Concatenating text columns: {', '.join(text_columns)}")
    df[concatenated_column] = df[text_columns].apply(
        lambda row: "\n".join(row.dropna().astype(str)), axis=1
    )
    print(f"Created concatenated text column: '{concatenated_column}'")

    df = await transform_text(
        df=df,
        text_column=concatenated_column,
        prompt=transformation_prompt,
        api_key=api_key,
        output_column="transformed_text",
        model=model,
        max_concurrent=max_concurrent_transforms,
    )

    df = await generate_embeddings(
        df=df,
        text_column="transformed_text",
        api_key=api_key,
        output_column="embedding",
        embedding_model=embedding_model,
        batch_size=embedding_batch_size,
        max_concurrent=max_concurrent_embeddings,
    )

    if dimensionality_methods is None:
        dimensionality_methods = [
            {"method": "pca", "n_components": [2, 3, 10]},
            {"method": "tsne", "n_components": [2, 3]},
            {"method": "umap", "n_components": [2, 3]},
        ]

    print(f"\nSaving results to {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Complete")

    return df


async def main():
    """Example usage."""
    import os

    INPUT_CSV = "courses.csv"
    OUTPUT_CSV = "courses_processed.csv"
    TEXT_COLUMNS = ["title", "subject_codes", "description"]

    TRANSFORMATION_PROMPT = """Rewrite this course description as a content-rich summary (50-100 words) that:
- Lists all key topics covered
- Uses clear, concise, explicit language
- Focuses on what is studied, not how it's taught
- Do not include logistical course details

Course description:
{text}

Summary:"""

    API_KEY = os.getenv("OPENAI_API_KEY")

    DIM_METHODS = [
        {"method": "pca", "n_components": [2, 3]},
        {"method": "tsne", "n_components": [2, 3]},
        {"method": "umap", "n_components": [2, 3]},
    ]

    RANDOM_SAMPLE = None

    if not API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        return

    await process_csv(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        text_columns=TEXT_COLUMNS,
        transformation_prompt=TRANSFORMATION_PROMPT,
        api_key=API_KEY,
        model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        dimensionality_methods=DIM_METHODS,
        random_sample=RANDOM_SAMPLE,
        max_concurrent_transforms=5,
        max_concurrent_embeddings=5,
        embedding_batch_size=100,
    )


if __name__ == "__main__":
    asyncio.run(main())
