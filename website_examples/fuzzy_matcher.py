import pandas as pd
from tfidf_matcher.ngrams import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import gradio as gr

def matcher(original=[], lookup=[], outname='Original', ngram_length=3, cutoff=0.8):
    k_matches=1

    # Enforce listtype, set to lower
    original = list(original.split(","))
    lookup = list(lookup.split(","))

    # print(original)
    # print(lookup)

    original_lower = [x.lower() for x in original]
    lookup_lower = [x.lower() for x in lookup]

    # Set ngram length for TfidfVectorizer callable
    def ngrams_user(string, n=ngram_length):
        return ngrams(string, n)

    # Generate Sparse TFIDF matrix from Lookup corpus
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_user)
    tf_idf_lookup = vectorizer.fit_transform(lookup_lower)

    # Fit KNN model to sparse TFIDF matrix generated from Lookup
    nbrs = NearestNeighbors(n_neighbors=k_matches, n_jobs=-1, metric="cosine").fit(tf_idf_lookup)

    # Use nbrs model to obtain nearest matches in lookup dataset. Vectorize first.
    tf_idf_original = vectorizer.transform(original_lower)
    distances, lookup_indices = nbrs.kneighbors(tf_idf_original)

    # Extract top Match Score (which is just the distance to the nearest neighbour),
    # Original match item, and Lookup matches.
    original_name_list = []
    confidence_list = []
    index_list = []
    lookup_list = []
    print(len(lookup_indices))
    # i is 0:len(original), j is list of lists of matches
    for i, lookup_index in enumerate(lookup_indices):
        original_name = original[i]
        # lookup names in lookup list
        lookups = [lookup[index] for index in lookup_index]
        # transform distances to confidences and store
        confidence = [1 - round(dist, 2) for dist in distances[i]]
        original_name_list.append(original_name)
        # store index
        index_list.append(lookup_index)
        confidence_list.append(confidence)
        lookup_list.append(lookups)

    # Convert to df
    df_orig_name = pd.DataFrame(original_name_list, columns=[outname])

    df_lookups = pd.DataFrame(
        lookup_list, columns=["Match"]
    )
    df_confidence = pd.DataFrame(
        confidence_list,
        columns=["Match Confidence"],
    )

    # bind columns
    matches = pd.concat([df_orig_name, df_lookups, df_confidence], axis=1)

    # reorder columns | can be skipped
    lookup_cols = list(matches.columns.values)
    lookup_cols_reordered = [lookup_cols[0]]
    for i in range(1, k_matches + 1):
        lookup_cols_reordered.append(lookup_cols[i])
        lookup_cols_reordered.append(lookup_cols[i + k_matches])
        # lookup_cols_reordered.append(lookup_cols[i + 2 * k_matches])
    matches = matches[lookup_cols_reordered]

    matches = matches.loc[matches["Match Confidence"] > cutoff]
    matches.sort_values(by=["Match Confidence"], ascending=False, inplace=True)

    return matches

def combine(a, b):
    return a + " " + b


with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Input a list of names", value='Courtney Walsh,Curtly Ambrose,Malcolm Marshall,Brian Lara,Viv Richards,Obama',lines=2)
            txt_2 = gr.Textbox(label="Input some names to match", value="Walsh, Ambrose, Marshall, Lara",lines=2)

    # with gr.Row():
        with gr.Column():

            outty =  gr.Dataframe(
                    headers=["Original", "Match", "Confidence"],
                    datatype=["str", "str", "number"],
                    label="Matched",
                )


    btn = gr.Button(value="Submit")
    btn.click(matcher, inputs=[txt, txt_2], outputs=[outty])



if __name__ == "__main__":
    demo.launch()