import argparse
import os
import json
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from scipy.stats import rankdata
from collections import Counter
import krippendorff

# Filenames to load from each model folder
def get_model_filenames():
    return [
        "gpt-4o-audio-preview-2024-12-17/emergent-tts-eval_strong-prompting_ballad_evaluation-predictions.jsonl",
        "HumeAI/emergent-tts-eval_strong-prompting_evaluation-predictions.jsonl",
        "eleven_multilingual_v2/emergent-tts-eval_nPczCjzI2devNBz1zQrb_evaluation-predictions.jsonl",
        "deepgram/emergent-tts-eval_thalia-en_evaluation-predictions.jsonl",
        "orpheus-tts-0.1-finetune-prod/emergent-tts-eval_tara_evaluation-predictions.jsonl",
        "Sesame1B/emergent-tts-eval_evaluation-predictions.jsonl",
        "Qwen2.5-Omni-7B/emergent-tts-eval_strong-prompting_Chelsie_evaluation-predictions.jsonl",
        "gpt-4o-mini-tts/emergent-tts-eval_strong-prompting_alloy_evaluation-predictions.jsonl",
    ]

FILEMNAME_2_MODELNAME = {
    "./EmergentTTSEvalResults/full_3fe115352376ebae9107dd79d3781edd": "Gemini 2.5 Pro",
    "./EmergentTTSEvalResults/full_3fe115352376ebae9107dd79d3781edd_judger_gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "./EmergentTTSEvalResults/full_3fe115352376ebae9107dd79d3781edd_judger_gemini-2.5-flash-preview-04-17": "Gemini 2.5 Flash",
    "./EmergentTTSEvalResults/full_3fe115352376ebae9107dd79d3781edd_judger_gpt-4o-audio-preview-2024-12-17": "gpt-4o-audio",
    "./EmergentTTSEvalResults/full_3fe115352376ebae9107dd79d3781edd_judger_gpt-4o-mini-audio-preview-2024-12-17": "gpt-4o-mini-audio",
    "./EmergentTTSEvalResults/full_3fe115352376ebae9107dd79d3781edd_judger_Qwen2.5-Omni-7B": "Qwen 2.5 Omni",
}

# parse_fn to extract model identifier from a path
MODEL_KEYS = [
    'gpt-4o-audio-preview-2024-12-17', 'HumeAI', 'eleven_multilingual_v2', 'deepgram',
    'orpheus-tts-0.1-finetune-prod', 'Sesame1B', 'Qwen2.5-Omni-7B', 'gpt-4o-mini-tts'
]

MODEL_JUDGES = [
    'gemini-2.5-pro-preview-05-06', 'gemini-2.0-flash-001', 'gemini-2.5-flash-preview-04-17', 
    'gpt-4o-audio-preview-2024-12-17', 'gpt-4o-mini-audio-preview-2024-12-17', 'Qwen2.5-Omni-7B'
]

PRETTY_MODEL_NAME = {
    'gpt-4o-audio-preview-2024-12-17': 'gpt-4o-audio', 
    'HumeAI': 'Hume AI', 
    'eleven_multilingual_v2': 'ElevenLabs', 
    'deepgram': 'DeepGram',
    'orpheus-tts-0.1-finetune-prod': 'Orpheus-TTS Tara', 
    'Sesame1B': 'Sesame1B', 
    'Qwen2.5-Omni-7B': 'Qwen 2.5 Omni Chelsie', 
    'gpt-4o-mini-tts': 'gpt-4o-mini-tts (SP)'
}

def pretty_name(model_name):
    return PRETTY_MODEL_NAME.get(model_name, model_name)

def parse_fn(path):
    if "4o_mini_tts" in path:       # hacky fix for gpt-4o-mini-tts
        return "gpt-4o-mini-tts"
    for key in MODEL_KEYS:
        if key in path:
            return key
    raise ValueError(f"Unknown model key in path: {path}")

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {path} at line {i}: {e}")
                print(f"Offending line ({i}): {line.strip()}")
                raise

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {path}: {e}")
            raise

def merge_and_save_evals(args):
    records = []

    # Load human evals
    for human_file in args.human_jsonls:
        if not os.path.isfile(human_file):
            raise FileNotFoundError(f"Human eval file not found: {human_file}")
        
        record_length = len(records)    # for assert

        human_eval = load_json(human_file)
        username = human_eval["user"]["name"]

        for obj in human_eval["responses"]:
            q = obj["question"]
            predicted = q["predicted_speech_index"]
            choice = obj["choice"]

            # determine winner for human
            if choice == "Audio A and B are equal":
                winner = 1  # tie
            elif (predicted == 1 and choice == "Audio A is better") or (predicted == 2 and choice == "Audio B is better"):
                winner = 2  # other model wins
            else:
                winner = 0  # baseline model wins

            records.append({
                'unique_id_eval': q["unique_id_eval"],
                'question_idx': obj["question_idx"],
                'question_timestamp': obj["timestamp_utc"],
                'rater': username,
                'evolution_depth': q["evolution_depth"],
                'category': q["category"],
                'foreign_language': q.get('misc', {}).get('foreign_language'),
                'baseline_model': parse_fn(q["baseline_audio_path"]),
                'other_model': parse_fn(q["audio_out_path"]),
                'baseline_score': None,
                'other_score': None,
                'winner': winner,
                'rater_rationale': obj["rationale"],
            })

        assert len(records) - record_length == len(human_eval["responses"])     # assert all entries in file are added

    if args.eval == "winrate":
        model_folders = args.model_folder     # expects only 1 model folder passed to command line
    elif args.eval == "human_model_agreement":
        model_folders = args.model_folder       # expects a list of model folders passed to command line
    else:   # perform human eval, do not expect args.model_foldera 
        model_folders = None

    if model_folders:
        for model_folder in model_folders:
            # Load model evals
            if not os.path.isdir(model_folder):
                raise NotADirectoryError(f"Model folder not found: {model_folder}")
            for fname in get_model_filenames():
                path = os.path.join(model_folder, fname)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Model eval file not found: {path}")
                
                record_length = len(records)    # for assert

                for obj in load_jsonl(path):
                    pred_idx = obj["predicted_speech_index"]
                    j = obj["judger_output_win_rate_based"]

                    # assign scores based on predicted index
                    if pred_idx == 1:
                        baseline_score = j.get("score_2")   # may encounter cannot parse score
                        other_score = j.get("score_1")
                        
                    else:
                        baseline_score = j.get("score_1")
                        other_score = j.get("score_2")

                    # determine winner for model eval
                    jwin = j["winner"]
                    if jwin == 0:   
                        winner = 1  # tie
                    elif jwin == -1:
                        winner = None   # cannot parse winner
                    elif (pred_idx == 1 and jwin == 1) or (pred_idx == 2 and jwin == 2):
                        winner = 2  # other model wins
                    else:
                        winner = 0 # baseline model wins

                    judger_model = obj["judger_model"]
                    assert judger_model in MODEL_JUDGES, f"Unknown model key: {judger_model}"

                    records.append({
                        'unique_id_eval': obj["unique_id_eval"],
                        'question_idx': None,
                        'question_timestamp': None,
                        'rater': judger_model,
                        'evolution_depth': obj["evolution_depth"],
                        'category': obj["category"],
                        'foreign_language': obj.get('misc', {}).get('foreign_language'),
                        'baseline_model': parse_fn(obj["baseline_audio_path"]),
                        'other_model': parse_fn(obj["audio_out_path"]),
                        'baseline_score': baseline_score,
                        'other_score': other_score,
                        'winner': winner,
                        'rater_rationale': j["system_comparison"],
                    })
                
                assert len(records) - record_length == 1645     # assert all entries in file are added 

    # Create DataFrame and output
    df = pd.DataFrame.from_records(records)
    print(df.head())

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "human_model_evals.csv")
    df.to_csv(output_csv, index=False)
    print(f"Merged DataFrame saved to {output_csv}")

    return df


# Eval 1.1: Win-rate table

def split_by_rater_type(df: pd.DataFrame):
    """
    Returns two DataFrames: (human_df, model_df),
    by checking `rater` against MODEL_JUDGES.
    """
    is_model = df['rater'].isin(MODEL_JUDGES)
    model_df = df[is_model].copy()
    human_df = df[~is_model].copy()
    print("Total rows:", len(df))
    print("→ human raters:", len(human_df))
    print("→ model raters:", len(model_df))
    return human_df, model_df

def compute_rate_and_ci(grp: pd.DataFrame):
    """
    Given grp with winner ∈ {0,1,2}, None dropped,
    returns (rate, ci_half_width).
    rate = (other_wins + 0.5*ties) / n
    ci = 1.96 * sqrt(p(1-p)/n)
    """
    grp = grp.dropna(subset=['winner'])
    n = len(grp)
    if n == 0:
        return float('nan'), float('nan')
    other = (grp['winner'] == 2).sum()
    ties  = (grp['winner'] == 1).sum()
    p_hat = (other + 0.5 * ties) / n
    se    = math.sqrt(p_hat * (1 - p_hat) / n)
    ci    = 1.96 * se   # two-sided 95% CI
    return p_hat, ci

def make_winrate_table(args, df: pd.DataFrame):
    """
    Builds and returns a DataFrame of shape (2, N_pairs),
    index=['human','model'], columns = sorted list of "baseline→other" keys.
    """
    # 1) split
    human_df, model_df = split_by_rater_type(df)
    # 2) identify all distinct (baseline, other) pairs in the full df
    pairs = (
        df[['baseline_model','other_model']]
        .drop_duplicates()
        .apply(lambda row: (row['baseline_model'], row['other_model']), axis=1)
        .tolist()
    )

    print("All pairs found:", pairs)
    print("Count of distinct pairs = ", len(pairs))
    assert len(pairs) == 8, f"Expected 8 model pairs but found {len(pairs)}"

    # sort for consistency
    pairs = sorted(pairs)
    col_labels = [f"{b}→{o}" for b, o in pairs]

    # 3) for each subset and each pair, compute win-rate
    # data = {}
    rate_data = {'human': [], 'model': []}
    ci_data   = {'human': [], 'model': []}
    for label, subset in [('human', human_df), ('model', model_df)]:
        for (b, o) in pairs:
            grp = subset[
                (subset['baseline_model'] == b) &
                (subset['other_model']    == o)
            ].dropna(subset=['winner'])

            rate, ci = compute_rate_and_ci(grp)
            rate_data[label].append(rate)
            ci_data[label].append(ci)

            print(f"{label} | {b}→{o}: total={len(grp)}, other_wins={(grp['winner'] == 2).sum()}, ties={(grp['winner'] == 1).sum()}, rate={rate:.2%} ± {ci:.2%}")

    # 4) assemble into a DataFrame
    rates = pd.DataFrame(rate_data, index=col_labels).T
    cis   = pd.DataFrame(ci_data,   index=col_labels).T

    save_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.model_folder[0])))
    os.makedirs(save_dir, exist_ok=True)
    rates.to_csv(os.path.join(save_dir, "winrates.csv"), index=True, float_format="%.4f")
    cis.to_csv(os.path.join(save_dir, "winrate_cis.csv"), index=True, float_format="%.4f")

    print("Saved win-rate table → winrates.csv")
    print("Saved 95% CI table → winrate_cis.csv")

    return rates, cis

def plot_winrate_bar(args, rates: pd.DataFrame, cis: pd.DataFrame):
    """
    Creates and saves a grouped bar plot of win-rates with 95% CIs.
    
    Parameters:
    - rates: DataFrame index=['human','model'], columns=['baseline→other', ...]
    - cis:   DataFrame same shape, containing 95% CI half-widths
    """
    # Extract the "other" model names for x-axis labels
    model_labels = [col.split('→')[1] for col in rates.columns]
    model_labels = [pretty_name(label) for label in model_labels]

    # Bar positions
    x = np.arange(len(model_labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    ax.bar(
        x - width/2,
        rates.loc['human'],
        width,
        yerr=cis.loc['human'],
        label='Human'
    )
    ax.bar(
        x + width/2,
        rates.loc['model'],
        width,
        yerr=cis.loc['model'],
        label=FILEMNAME_2_MODELNAME.get(args.model_folder[0], "Model judge")
    )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Comparison TTS Models', fontsize=13)
    ax.set_ylabel('Win Rate', fontsize=13)
    ax.legend()
    plt.tight_layout()

    # Ensure output directory exists
    save_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.model_folder[0])))
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "winrate_barplot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved bar plot to {plot_path}")

def compute_ranking_agreement(args, rates: pd.DataFrame):
    """
    Given `rates` with index ['human','model'] and columns ['baseline→other', ...],
    computes:
      - human_rank: list of other-model names sorted by human win-rate desc
      - model_rank: list of other-model names sorted by model win-rate desc
      - spearman: Spearman rho between the two rankings
    and saves all three into one JSON: ranking_summary.json
    """
    # pull out the two series
    human_rates = rates.loc['human']
    model_rates = rates.loc['model']
    
    # build rank order lists of "other" model names
    human_rank = [col.split('→',1)[1] for col in human_rates.sort_values(ascending=False).index]
    model_rank = [col.split('→',1)[1] for col in model_rates.sort_values(ascending=False).index]
    
    # compute Spearman rho
    hr = human_rates.rank(ascending=False, method='min')
    mr = model_rates.rank(ascending=False, method='min')
    spearman = hr.corr(mr)  # Pearson of rank vectors = Spearman
    
    # prepare the summary dict
    summary = {
        "human_rank": human_rank,
        "model_rank": model_rank,
        "spearman": spearman
    }
    
    # ensure output dir
    save_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.model_folder[0])))
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "ranking_summary.json")
    
    # save combined JSON
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved ranking summary → {out_path}")
    print(f"Spearman correlation: {spearman:.3f}")
    
    return human_rank, model_rank, spearman


# Eval 1.2: Human-model judge agreement table

def kendalls_w(ratings: np.ndarray) -> float:
    """
    ratings: array shape (n_items, m_raters)
    returns: Kendall's W
    """
    n, m = ratings.shape
    # rank within each column (rater)
    ranks = np.apply_along_axis(rankdata, 0, ratings)
    # sum of ranks per item
    R = np.sum(ranks, axis=1)
    S = np.sum((R - R.mean())**2)
    W = 12 * S / (m**2 * (n**3 - n))
    return W

def compute_human_model_agreement_single_comp(args, df: pd.DataFrame) -> pd.DataFrame:
    human_df, model_df = split_by_rater_type(df)
    results = {}

    for other in sorted(df['other_model'].unique()):
        h_sub = (
            human_df[human_df['other_model'] == other]
            [['unique_id_eval', 'winner', 'rater']]
            .dropna(subset=['winner'])
        )
        m_sub = (
            model_df[model_df['other_model'] == other]
            [['unique_id_eval', 'winner', 'rater']]
            .dropna(subset=['winner'])
            .rename(columns={'winner': 'winner_m'})
        )

        # 1) inner join without sorting
        merged = pd.merge(
            h_sub, m_sub,
            on='unique_id_eval',
            how='inner',
            sort=False         # <-- turn off pandas’ auto-sort
        )

        print(merged.head())

        # 2) now sort by the key to align both columns
        # merged = merged.sort_values('unique_id_eval')

        # 3) pull out the aligned winner vectors
        h_list = merged['winner'].astype(int).to_numpy()
        m_list = merged['winner_m'].astype(int).to_numpy()

        diffs = np.abs(h_list - m_list)          # array of absolute differences
        dist_counts = Counter(diffs)             # count how many times each distance occurs

        print("Distance → Count")
        for distance, count in sorted(dist_counts.items()):
            print(f"{distance:>8} → {count}")

        assert len(h_list) == len(m_list), (
            f"Still a mismatch for {other}: {len(h_list)} vs {len(m_list)}"
        )
        print(f"{other}: aligned {len(h_list)} items")

        # compute the three metrics
        kappa  = cohen_kappa_score(h_list, m_list)
        wkappa = cohen_kappa_score(h_list, m_list, weights='quadratic')
        w      = kendalls_w(np.vstack([h_list, m_list]).T)

        results[other] = {
            'cohen_kappa':    kappa,
            'weighted_kappa': wkappa,
            'kendall_W':      w
        }

    agreement_df = pd.DataFrame.from_dict(results, orient='index')

    # Save & print
    save_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.model_folder[0])))
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, "human_model_agreement.csv")
    agreement_df.to_csv(out_csv, float_format="%.4f")

    print("Agreement metrics (rows=other_model):")
    print(agreement_df)
    print(f"Saved human vs model agreement → {out_csv}")

    return agreement_df


# Eval 2: Human inter-rater agreement

def compute_human_human_overlap(args, df: pd.DataFrame) -> pd.DataFrame:
    # extract only human ratings
    human_df, _ = split_by_rater_type(df)
    
    # drop duplicates so each (rater, unique_eval_id, other_model) is only counted once
    tasks = (
        human_df[['rater', 'unique_id_eval', 'other_model', 'winner']]
        .dropna(subset=['winner'])
        [['rater', 'unique_id_eval', 'other_model']]
        .drop_duplicates()
    )

    assert len(tasks) == len(human_df)
    
    # get sorted list of all raters
    raters = sorted(tasks['rater'].unique())
    
    # initialize an all-zero matrix
    matrix = pd.DataFrame(
        0,
        index=raters,
        columns=raters,
        dtype=int
    )
    
    # for each distinct task, look at which raters rated it—
    # then increment every pairwise combination (including self)
    grouped = tasks.groupby(
        ['unique_id_eval', 'other_model']
    )['rater'].unique()
    
    for rater_array in grouped:
        for i in rater_array:
            for j in rater_array:
                matrix.at[i, j] += 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "human_human_overlap.csv")
    matrix.to_csv(out_csv)
    print("Saved human–human overlap matrix →", out_csv)
    
    return matrix

def compute_human_human_agreement(args, df: pd.DataFrame, overlap_matrix: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    For every pair of human raters with >threshold overlapping tasks,
    compute Pearson’s r, Cohen’s κ, and quadratic‐weighted κ on their
    shared ratings.
    """
    # 1) grab only the human‐rated rows
    human_df, _ = split_by_rater_type(df)

    results = {}
    raters = overlap_matrix.index.tolist()

    # 2) loop over unique rater‐pairs
    for i, r1 in enumerate(raters):
        for r2 in raters[i+1:]:
            overlap = overlap_matrix.at[r1, r2]
            if overlap <= threshold:
                continue

            # 3) extract each rater's ratings
            h1 = (
                human_df[human_df['rater'] == r1]
                [['unique_id_eval', 'other_model', 'winner']]
                .dropna(subset=['winner'])
            )
            h2 = (
                human_df[human_df['rater'] == r2]
                [['unique_id_eval', 'other_model', 'winner']]
                .dropna(subset=['winner'])
                .rename(columns={'winner': 'winner_2'})
            )

            # 4) inner‐join to get only the tasks both rated
            merged = pd.merge(
                h1, h2,
                on=['unique_id_eval', 'other_model'],
                how='inner'
            )
            h_list  = merged['winner'].astype(int).to_numpy()
            h2_list = merged['winner_2'].astype(int).to_numpy()

            assert len(h_list) == len(h2_list)

            # 5) compute metrics
            pearson_r    = np.corrcoef(h_list, h2_list)[0,1]
            cohen_kappa  = cohen_kappa_score(h_list, h2_list)
            w_cohen_kappa = cohen_kappa_score(h_list, h2_list, weights='quadratic')

            results[(r1, r2)] = {
                'overlap':          overlap,
                'pearson_r':        pearson_r,
                'cohen_kappa':      cohen_kappa,
                'weighted_kappa':   w_cohen_kappa
            }

    # 6) build a MultiIndex DataFrame
    agreement_df = pd.DataFrame.from_dict(results, orient='index')
    agreement_df.index = pd.MultiIndex.from_tuples(
        agreement_df.index,
        names=['rater1', 'rater2']
    )

    # 7) save and return
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "human_human_agreement.csv")
    agreement_df.to_csv(out_csv, float_format="%.4f")
    print("Saved human–human agreement metrics →", out_csv)

    return agreement_df

def compute_alpha_overall_and_by_category(args, df: pd.DataFrame, level: str = 'ordinal'):
    """
    Compute Krippendorff’s α both overall and per category.

    Returns
    -------
    overall_alpha : float
        Alpha across all human ratings.
    alpha_by_cat : pd.Series
        Alpha for each category, sorted descending.
    """
    # 1) Filter human ratings and drop missing
    human_df, _ = split_by_rater_type(df)

    human_df_len = len(human_df)

    human_df = (
        human_df[['category', 'rater', 'unique_id_eval', 'other_model', 'winner']]
        .dropna(subset=['winner'])
        .drop_duplicates(subset=['rater','unique_id_eval','other_model'])
    )

    assert len(human_df) == human_df_len

    # Helper to pivot and compute alpha from a sub-dataframe
    def pivot_and_alpha(subdf):
        subdf = subdf.copy()
        subdf['task_id'] = subdf['unique_id_eval'].astype(str) + "|" + subdf['other_model']
        mat = subdf.pivot_table(
            index='rater',
            columns='task_id',
            values='winner',
            aggfunc='first'
        ).to_numpy()
        return krippendorff.alpha(reliability_data=mat, level_of_measurement=level)

    # 2) Overall alpha
    overall_alpha = pivot_and_alpha(human_df)
    print(f"Overall Krippendorff’s α ({level}): {overall_alpha:.4f}")

    # 3) Alpha by category
    alphas = {}
    for cat, group in human_df.groupby('category'):
        alphas[cat] = pivot_and_alpha(group)
    alpha_by_cat = pd.Series(alphas).sort_values(ascending=False)
    print("\nKrippendorff’s α by category:")
    print(alpha_by_cat)

    # 4) (Optional) Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "human_krippendorff_alpha.json")
    with open(out_path, 'w') as f:
        json.dump(
            {'krippendorff_alpha_overall': overall_alpha,
             'krippendorff_alpha_by_category': [
                 {'category': cat, 'alpha': float(alpha)} for cat, alpha in alpha_by_cat.items()
             ]
             }, f, indent=2)
    
    print(f"Saved Krippendorff’s α results → {out_path}")

    return overall_alpha, alpha_by_cat


# Eval 3: Human-model inter-rater agreement
def compute_human_model_overlap(args, df: pd.DataFrame) -> pd.DataFrame:
    """
    Count, for every pair of raters (human or model), how many distinct
    (unique_id_eval, other_model) tasks they both rated.
    """
    # 1) no split—take every (rater, unique_id_eval, other_model)
    tasks = (
        df[['rater', 'unique_id_eval', 'other_model']]
        .drop_duplicates()
    )

    assert len(tasks) == len(df)

    # 2) list of all raters (both humans and model‐judges)
    raters = sorted(tasks['rater'].unique())

    # 3) init zero matrix
    matrix = pd.DataFrame(
        0,
        index=raters,
        columns=raters,
        dtype=int
    )

    # 4) group by each task, collect its raters
    grouped = tasks.groupby(
        ['unique_id_eval', 'other_model']
    )['rater'].unique()

    # 5) for each task, bump every pair (i,j) that co-rated it
    for rater_array in grouped:
        for i in rater_array:
            for j in rater_array:
                matrix.at[i, j] += 1

    # 6) save & return
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "human_model_overlap.csv")
    matrix.to_csv(out_csv)
    print("Saved full human-model overlap matrix →", out_csv)

    return matrix

def compute_human_model_agreement(args, df: pd.DataFrame, overlap_matrix: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    For every pair of raters (human or model) with >threshold overlapping tasks,
    compute Pearson’s r, Cohen’s κ, and quadratic-weighted κ on their shared ratings.
    """
    results = {}
    raters = overlap_matrix.index.tolist()

    # 2) loop over unique rater-pairs
    for i, r1 in enumerate(raters):
        for r2 in raters[i+1:]:
            overlap = overlap_matrix.at[r1, r2]
            if overlap <= threshold:
                continue

            # 3) pull out each rater’s ratings
            r1_df = (
                df[df['rater'] == r1]
                [['unique_id_eval', 'other_model', 'winner']]
                .dropna(subset=['winner'])
            )
            r2_df = (
                df[df['rater'] == r2]
                [['unique_id_eval', 'other_model', 'winner']]
                .dropna(subset=['winner'])
                .rename(columns={'winner': 'winner_2'})
            )

            # 4) inner-join to align on the tasks both rated
            merged = pd.merge(
                r1_df, r2_df,
                on=['unique_id_eval', 'other_model'],
                how='inner'
            )

            h_list   = merged['winner'].astype(int).to_numpy()
            h2_list  = merged['winner_2'].astype(int).to_numpy()

            assert len(h_list) == len(h2_list), (
                f"Mismatch for ({r1},{r2}): {len(h_list)} vs {len(h2_list)}"
            )

            # 5) compute metrics
            pearson_r     = np.corrcoef(h_list, h2_list)[0,1]
            cohen_kappa   = cohen_kappa_score(h_list, h2_list)
            w_cohen_kappa = cohen_kappa_score(h_list, h2_list, weights='quadratic')

            results[(r1, r2)] = {
                'overlap':        overlap,
                'pearson_r':      pearson_r,
                'cohen_kappa':    cohen_kappa,
                'weighted_kappa': w_cohen_kappa
            }

    # 6) build a MultiIndex DataFrame
    agreement_df = pd.DataFrame.from_dict(results, orient='index')
    agreement_df.index = pd.MultiIndex.from_tuples(
        agreement_df.index,
        names=['rater1', 'rater2']
    )

    # 7) save & return
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "human_model_agreement.csv")
    agreement_df.to_csv(out_csv, float_format="%.4f")
    print("Saved rater agreement metrics →", out_csv)

    return agreement_df

def plot_human_model_agreement_matrix(args, agreement_df: pd.DataFrame, threshold: int):
    # 1) Build the full ordered rater list
    all_raters = sorted(
        set(agreement_df.index.get_level_values('rater1'))
        | set(agreement_df.index.get_level_values('rater2'))
    )
    # Models first (in MODEL_JUDGES order), then humans
    modelers = [r for r in MODEL_JUDGES if r in all_raters]
    humans   = sorted(r for r in all_raters if r not in MODEL_JUDGES)
    raters   = modelers + humans
    n = len(raters)
    idx_map = {r: i for i, r in enumerate(raters)}

    os.makedirs(args.output_dir, exist_ok=True)

    # 2) Make one matrix per metric
    for metric in ['pearson_r', 'cohen_kappa', 'weighted_kappa']:
        # initialize with NaN, diagonal=1
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0

        # fill valid entries
        for (r1, r2), row in agreement_df.iterrows():
            if row['overlap'] <= threshold:
                continue
            i, j = idx_map[r1], idx_map[r2]
            val = row[metric]
            mat[i, j] = val
            mat[j, i] = val

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad('black')  # mask missing below-threshold as black
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, vmin=-1 if metric=='pearson_r' else 0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_xticklabels(raters, rotation=90, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(raters, fontsize=8)
        ax.set_title(metric.replace('_', ' ').title())

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric)

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"human_model_agreement_{metric}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {metric} matrix → {out_path}")

def rank_raters_by_avg_corr(agreement_df: pd.DataFrame, metric: str = 'pearson_r') -> pd.Series:
    """
    Given agreement_df indexed by (rater1, rater2) and containing
    a column `metric` (e.g. 'pearson_r', 'cohen_kappa', 'weighted_kappa'), compute each rater’s
    average metric across all their pairings, and return a
    descendingly sorted Series.
    """
    # 1) bring the index into columns
    df = agreement_df.reset_index()[['rater1', 'rater2', metric]]
    
    # 2) gather each rater’s metric into one long column
    df1 = df[['rater1', metric]].rename(columns={'rater1':'rater'})
    df2 = df[['rater2', metric]].rename(columns={'rater2':'rater'})
    long = pd.concat([df1, df2], ignore_index=True)
    
    # 3) group and average
    avg_corr = long.groupby('rater')[metric].mean()
    
    # 4) sort descending
    return avg_corr.sort_values(ascending=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge human and model eval JSONL files into a single DataFrame"
    )
    parser.add_argument(
        '--human-jsonls', '-H', nargs='+', required=True,
        help="Paths to JSONL files containing human evaluations"
    )
    parser.add_argument(
        '--model-folder', '-M', nargs='+', required=False,
        help="Directory each containing the 8 model evaluation JSONLs"
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help="Optional dir to save outputs"
    )
    parser.add_argument(
        '--eval', '-E',
        choices=['winrate', 'human_agreement', "human_model_agreement"],
        required=True,
        help="Which evaluation to perform: 'winrate', 'human_agreement', or 'human_model_agreement'"
    )
    
    args = parser.parse_args()

    if args.eval == "winrate":
        # Eval 1: Win-rate table
        # 1. Load and merge human, model evaluations
        eval_df = merge_and_save_evals(args)

        # 1.1 Create win-rate table
        rates, cis = make_winrate_table(args, eval_df)
        # 1.2 Plot win-rate bar chart
        plot_winrate_bar(args, rates, cis)
        # 1.3 Compute ranking agreement
        compute_ranking_agreement(args, rates)
        # 1.4 Compute human-model preference agreement
        compute_human_model_agreement_single_comp(args, eval_df)

    elif args.eval == "human_agreement":
        # Eval 2: Human inter-rater agreement
        # 1. Load and merge human evaluations
        eval_df = merge_and_save_evals(args)
        overlap_matrix = compute_human_human_overlap(args, eval_df)
        print(overlap_matrix)

        agreement_matrix = compute_human_human_agreement(args, eval_df, overlap_matrix, 50)
        print(agreement_matrix)

        compute_alpha_overall_and_by_category(args, eval_df)

    elif args.eval == "human_model_agreement":
        # Eval 3: Human-model agreement
        # 1. Load and merge human, model evaluations
        eval_df = merge_and_save_evals(args)

        overlap_matrix = compute_human_model_overlap(args, eval_df)
        print(overlap_matrix)

        agreement_matrix = compute_human_model_agreement(args, eval_df, overlap_matrix, 50)
        print(agreement_matrix)

        plot_human_model_agreement_matrix(args, agreement_matrix, threshold=50)

        for m in ['pearson_r', 'cohen_kappa', 'weighted_kappa']:
            print(f"Ranking by {m}:")
            print(rank_raters_by_avg_corr(agreement_matrix, m))
            print("\n")

    elif args.eval == "human_model_category_agreement":
        # Eval 4: Human-model category agreement
        # 1. Load and merge human, model evaluations
        eval_df = merge_and_save_evals(args)

    else:
        raise ValueError(f"Unknown eval type: {args.eval}")

    print("Done")
