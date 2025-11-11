import pandas as pd
import numpy as np
import umap
import ast
import sys

def main():
    n_components = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    print(f"Computing UMAP ({n_components}D)...")
    
    df = pd.read_csv("./courses_processed.csv")
    embeddings = np.array([ast.literal_eval(emb) for emb in df['embedding'] if emb is not None])
    
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=100,
        metric='cosine',
        spread=1.0,
        low_memory=False
    )
    coords_umap = umap_model.fit_transform(embeddings)
    
    df[f'umap_{n_components}d'] = [str(coord.tolist()) for coord in coords_umap]
    df.to_csv("./courses_processed.csv", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()