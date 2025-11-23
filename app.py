# app.py — ULTRA-OPTIMIZED v2 + PERSISTENT PET DATABASE + ▲▼ REORDER (NO JS)
# ✅ Works flawlessly on Streamlit 1.32–1.51+
# ✅ Pure Streamlit (no JS, no fragments)
# ✅ Drag-free, robust pet reordering
# ✅ Same speed & correctness

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os

# ==============================
# NUMBA JIT SETUP
# ==============================
try:
    from numba import njit, prange, float32, int32, int8
    HAS_NUMBA = True
except ImportError:
    st.error("❌ Numba is required. Install via: `pip install numba`")
    st.stop()

# ==============================
# PET DATABASE SETUP
# ==============================
DB_PATH = "pets.json"

DEFAULT_PETS = {
    "Chicken": [1, 2, 3, 4, 5, 6],
    "Owl": [3, 3, 3, 3, 5, 5],
    "Snake": [4, 4, 6, 6, 8, 8],
    "White Tiger": [0, 2, 4, 6, 8, 10],
}

def load_pets():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Convert old format
                    pets_list = [{"name": name, "dice": dice} for name, dice in data.items()]
                elif isinstance(data, list):
                    pets_list = data
                else:
                    pets_list = []
                return pets_list
        except Exception as e:
            st.warning(f"⚠️ Failed to load {DB_PATH}, using defaults. Error: {e}")
    # Create default DB
    pets_list = [{"name": name, "dice": dice} for name, dice in DEFAULT_PETS.items()]
    save_pets(pets_list)
    return pets_list

def save_pets(pets_list):
    try:
        with open(DB_PATH, "w") as f:
            json.dump(pets_list, f, indent=2)
    except Exception as e:
        st.error(f"❌ Failed to save pets: {e}")

# ==============================
# CONSTANTS — TYPED FOR JIT
# ==============================
GOAL = np.int32(120)
BIAS_LEVELS = np.array([3, 5, 10, 20, 40], dtype=np.int32)
COLORS = ["red", "green", "blue", "gray"]

ITEM_TO_ID = {
    None: 0,
    "Cherry": 1,
    "Peach": 2,
    "Double dice": 3,
    "Single banana": 4,
    "Cluster banana": 5,
    "Sleep": 6,
}

CHERRY_ADD = np.array([0, 1, 2], dtype=np.int32)
PEACH_ADD  = np.array([2, 3, 4], dtype=np.int32)
SINGLE_SUB = np.array([2, 1, 0], dtype=np.int32)
CLUSTER_SUB= np.array([4, 3, 2], dtype=np.int32)

POS_ITEMS = ["Cherry", "Peach", "Double dice"]
NEG_ITEMS = ["Single banana", "Cluster banana", "Sleep"]


# ==============================
# CORE JIT — FULLY OPTIMIZED
# ==============================
@njit(fastmath=True)
def biased_choice_jit(options, values, bias):
    k = bias * 3.0
    n = len(options)
    weights = np.empty(n, dtype=np.float32)
    total = 0.0
    for i in range(n):
        v = max(values[i], 0.1)
        w = v ** k
        w = max(w, 1e-8)
        weights[i] = w
        total += w
    if total <= 0.0:
        return options[0]
    r = np.float32(np.random.random()) * total
    cum = 0.0
    for i in range(n):
        cum += weights[i]
        if r <= cum:
            return options[i]
    return options[-1]

@njit(fastmath=True)
def roll_pet_jit(pet_dice, unique_dice, item_id, bias, use_unique):
    bias_idx = 1
    if bias <= -0.5:
        bias_idx = 0
    elif bias >= 0.5:
        bias_idx = 2

    if use_unique:
        n_unique = 0
        for val in unique_dice:
            if val == -1:
                break
            n_unique += 1
        if n_unique == 0:
            base = np.int32(1)
        else:
            base = biased_choice_jit(unique_dice[:n_unique], unique_dice[:n_unique], bias)
    else:
        base = biased_choice_jit(pet_dice, pet_dice, bias)

    if item_id == 1:
        roll = base + CHERRY_ADD[bias_idx]
    elif item_id == 2:
        roll = base + PEACH_ADD[bias_idx]
    elif item_id == 4:
        roll = base - SINGLE_SUB[bias_idx]
    elif item_id == 5:
        roll = base - CLUSTER_SUB[bias_idx]
    elif item_id == 6:
        roll = 1
    else:
        roll = base
    return roll if roll > 0 else 0

@njit(fastmath=True)
def simulate_one_race_jit(
    pet_dices, unique_dices, item_ids_flat, item_counts, biases, use_unique, deck_indices
):
    pos = np.zeros(4, dtype=np.int32)
    ptr = np.zeros(4, dtype=np.int32)
    while True:
        for i in range(4):
            if pos[i] >= GOAL: continue
            item_id = 0
            if ptr[i] < item_counts[i]:
                idx = deck_indices[i, ptr[i]]
                item_id = item_ids_flat[i, idx]
                ptr[i] += 1
            roll = roll_pet_jit(pet_dices[i], unique_dices[i], item_id, biases[i], use_unique)
            pos[i] += roll

        finishers = np.empty(4, dtype=np.int8)
        n_finish = 0
        for i in range(4):
            if pos[i] >= GOAL:
                finishers[n_finish] = i
                n_finish += 1
        if n_finish == 0: continue
        if n_finish == 1: return finishers[0]

        while True:
            max_pos = -1
            for j in range(n_finish):
                i = finishers[j]
                if pos[i] > max_pos: max_pos = pos[i]
            leaders = np.empty(4, dtype=np.int8)
            n_lead = 0
            for j in range(n_finish):
                i = finishers[j]
                if pos[i] == max_pos:
                    leaders[n_lead] = i
                    n_lead += 1
            if n_lead == 1: return leaders[0]
            for j in range(n_lead):
                i = leaders[j]
                item_id = 0
                if ptr[i] < item_counts[i]:
                    idx = deck_indices[i, ptr[i]]
                    item_id = item_ids_flat[i, idx]
                    ptr[i] += 1
                roll = roll_pet_jit(pet_dices[i], unique_dices[i], item_id, biases[i], use_unique)
                pos[i] += roll

@njit(parallel=True, fastmath=True)
def run_n_simulations_jit(n_sims, pet_dices, unique_dices, item_ids_flat, item_counts, biases, use_unique, all_decks):
    wins = np.zeros(4, dtype=np.int32)
    for sim_idx in prange(n_sims):
        deck = all_decks[sim_idx]
        winner = simulate_one_race_jit(pet_dices, unique_dices, item_ids_flat, item_counts, biases, use_unique, deck)
        wins[winner] += 1
    return wins

# ==============================
# PYTHON HELPERS
# ==============================
def prep_inputs(pet_dices, pet_items_list):
    pet_dices_arr = np.array(pet_dices, dtype=np.int32)
    unique_dices = np.full((4, 6), -1, dtype=np.int32)
    for i, dice in enumerate(pet_dices):
        uniq = sorted(set(dice))
        unique_dices[i, :len(uniq)] = uniq
    max_items = max(len(items) for items in pet_items_list) if pet_items_list else 0
    item_counts = np.array([len(items) for items in pet_items_list], dtype=np.int32)
    item_ids_flat = np.zeros((4, max_items), dtype=np.int8)
    for i, items in enumerate(pet_items_list):
        ids = [ITEM_TO_ID[item] for item in items]
        item_ids_flat[i, :len(ids)] = ids
    return pet_dices_arr, unique_dices, item_ids_flat, item_counts

def generate_all_decks(n_sims, pet_items_list):
    max_items = max(len(items) for items in pet_items_list) if pet_items_list else 0
    decks = np.zeros((n_sims, 4, max_items), dtype=np.int16)
    rng = np.random.default_rng(42)
    for pet_i, items in enumerate(pet_items_list):
        L = len(items)
        if L == 0: continue
        base = np.arange(L, dtype=np.int16)
        perms = rng.permuted(np.broadcast_to(base, (n_sims, L)), axis=1)
        decks[:, pet_i, :L] = perms
    return decks

def run_simulations(pet_dices, pet_items_list, pet_biases, use_unique=False, n_sims=30000):
    pet_dices_arr, unique_dices, item_ids_flat, item_counts = prep_inputs(pet_dices, pet_items_list)
    biases_arr = np.array(pet_biases, dtype=np.float32)
    all_decks = generate_all_decks(n_sims, pet_items_list)
    wins = run_n_simulations_jit(n_sims, pet_dices_arr, unique_dices, item_ids_flat, item_counts, biases_arr, use_unique, all_decks)
    total = wins.sum()
    return (wins / total).tolist() if total > 0 else [0.25] * 4

# ==============================
# STREAMLIT — ROBUST PET MANAGEMENT
# ==============================
def main():
    st.set_page_config(page_title="Pet Racing Rigging Analyzer", layout="wide")
    st.title("🐾 Pet Racing Win Rate vs Bias Strength")
    st.caption("V1 (face dice) vs V2 (unique dice). Fair (0%) and biased scenarios.")

    pets_list = load_pets()
    pets_dict = {p["name"]: p["dice"] for p in pets_list}
    pet_names_ordered = [p["name"] for p in pets_list]

    # Sidebar
    with st.sidebar:
        st.header("🐶 Pets (reorder with ▲ ▼)")

        # Reordering UI — pure Streamlit
        if len(pets_list) > 1:
            st.caption("Click ▲/▼ to move a pet up/down in the list. (This is done because drag function is broken in streamlit.)")
            for idx, pet in enumerate(pets_list):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
                col1.write(f"`{pet['name']}`: {pet['dice']}")
                # Up button (not for first)
                if idx > 0 and col2.button("▲", key=f"up_{idx}"):
                    pets_list[idx], pets_list[idx-1] = pets_list[idx-1], pets_list[idx]
                    save_pets(pets_list)
                    st.rerun()
                # Down button (not for last)
                if idx < len(pets_list) - 1 and col3.button("▼", key=f"down_{idx}"):
                    pets_list[idx], pets_list[idx+1] = pets_list[idx+1], pets_list[idx]
                    save_pets(pets_list)
                    st.rerun()
                # Delete
                if col4.button("🗑", key=f"del_btn_{idx}"):
                    del pets_list[idx]
                    save_pets(pets_list)
                    st.rerun()
        elif pets_list:
            st.write(f"`{pets_list[0]['name']}`: {pets_list[0]['dice']}")
            if st.button("🗑 Delete", key="del_only"):
                pets_list.clear()
                save_pets(pets_list)
                st.rerun()
        else:
            st.info("No pets. Add one below.")

        with st.expander("➕ Add Pet"):
            name = st.text_input("Pet Name", key="new_pet_name")
            dice_str = st.text_input("Dice (6 comma-separated ints)", "1,2,3,4,5,6", key="new_pet_dice")
            if st.button("💾 Save Pet"):
                try:
                    dice = [int(x.strip()) for x in dice_str.split(",")]
                    if len(dice) != 6:
                        raise ValueError("Exactly 6 dice values required")
                    # Add to end
                    pets_list.append({"name": name, "dice": dice})
                    save_pets(pets_list)
                    st.success(f"✅ Added {name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

        manual_mode = st.checkbox("🎛 Manual Input (no auto-save)", False)

    # Refresh after any sidebar change
    pets_dict = {p["name"]: p["dice"] for p in pets_list}
    pet_names_ordered = [p["name"] for p in pets_list]

    # Race setup
    st.header("🏁 Race Configuration")
    cols = st.columns(4)
    pet_names, pet_dices, pet_items_list = [], [], []
    new_manual_pets = {}

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"#### Pet {i+1}")
            if manual_mode:
                name = st.text_input(f"Name", f"Pet {i+1}", key=f"n{i}")
                dice_in = st.text_input(f"Dice", "1,2,3,4,5,6", key=f"d{i}")
                try:
                    dice = [int(x.strip()) for x in dice_in.split(",")]
                    if len(dice) != 6:
                        raise ValueError
                except:
                    dice = [1, 2, 3, 4, 5, 6]
                if name not in pets_dict and name.strip() and name != f"Pet {i+1}":
                    new_manual_pets[name] = dice
            else:
                if not pet_names_ordered:
                    st.error("⚠️ Add pets in sidebar first!")
                    return
                name = st.selectbox("Pet", pet_names_ordered, key=f"p{i}")
                dice = pets_dict[name]
            pet_names.append(name)
            pet_dices.append(dice)

            st.caption("Items")
            pos_cols = st.columns(3)
            neg_cols = st.columns(3)
            items = {}
            for j, it in enumerate(POS_ITEMS):
                items[it] = pos_cols[j].number_input(it, 0, 9999, 0, key=f"pos{i}{j}")
            for j, it in enumerate(NEG_ITEMS):
                items[it] = neg_cols[j].number_input(it, 0, 9999, 0, key=f"neg{i}{j}")
            item_list = [item for item, cnt in items.items() for _ in range(cnt)]
            pet_items_list.append(item_list)

    st.header("⚙️ Simulation")
    n_sims = st.number_input("Simulations per (model × scenario × bias)", 2000, 5000000, 30000, 1000)
    run = st.button("🚀 Run All Simulations", type="primary")

    if run:
        if new_manual_pets and manual_mode:
            st.info("🆕 New pets in Manual Mode")
            save_new = st.checkbox("✅ Save new pets to database", True, key="save_new_pets")
            if save_new:
                for name, dice in new_manual_pets.items():
                    if name not in pets_dict:
                        pets_list.append({"name": name, "dice": dice})
                save_pets(pets_list)
                st.success(f"💾 Saved {len(new_manual_pets)} new pets!")

        start_time = time.time()
        status = st.empty()
        status.info("Step 1: Fair win rates (V1 & V2)...")

        fair_v1 = run_simulations(pet_dices, pet_items_list, [0,0,0,0], use_unique=False, n_sims=n_sims)
        fair_v2 = run_simulations(pet_dices, pet_items_list, [0,0,0,0], use_unique=True, n_sims=n_sims)

        ranked_idx = np.argsort(fair_v1)[::-1]
        top_i, second_i, third_i = ranked_idx[0], ranked_idx[1], ranked_idx[2]
        top_name, second_name, third_name = pet_names[top_i], pet_names[second_i], pet_names[third_i]

        scenarios_bias = {
            "Punish Top": {b: [b/100 if i == top_i else 0 for i in range(4)] 
                          for b in [0] + [-x for x in BIAS_LEVELS]},
            "Boost 2nd": {b: [b/100 if i == second_i else 0 for i in range(4)] 
                         for b in [0] + BIAS_LEVELS.tolist()},
            "Boost 3rd": {b: [b/100 if i == third_i else 0 for i in range(4)] 
                         for b in [0] + BIAS_LEVELS.tolist()},
        }

        results = {"Fair": {"V1": fair_v1, "V2": fair_v2}}
        total_jobs = 3 * 2 * len(BIAS_LEVELS)
        done = 0

        for scenario, bias_dict in scenarios_bias.items():
            results[scenario] = {"V1": {}, "V2": {}}
            results[scenario]["V1"][0] = fair_v1.copy()
            results[scenario]["V2"][0] = fair_v2.copy()
            non_zero_biases = [b for b in bias_dict.keys() if b != 0]
            for bias_val in non_zero_biases:
                bias_vec = bias_dict[bias_val]
                done += 1
                status.info(f"Running: {scenario}, V1, {bias_val:+}% ({done}/{total_jobs})")
                results[scenario]["V1"][bias_val] = run_simulations(pet_dices, pet_items_list, bias_vec, use_unique=False, n_sims=n_sims)
                done += 1
                status.info(f"Running: {scenario}, V2, {bias_val:+}% ({done}/{total_jobs})")
                results[scenario]["V2"][bias_val] = run_simulations(pet_dices, pet_items_list, bias_vec, use_unique=True, n_sims=n_sims)

        elapsed = time.time() - start_time
        status.success(f"✅ Done in {elapsed:.1f}s!")

        st.subheader("📊 Fair Win Rates (0% Bias)")
        fair_df = pd.DataFrame({
            "Pet": pet_names,
            "V1 (Face)": [f"{r:.1%}" for r in fair_v1],
            "V2 (Unique)": [f"{r:.1%}" for r in fair_v2]
        }).set_index("Pet")
        st.table(fair_df)

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        ax0 = axes[0]
        x = np.arange(4)
        width = 0.35
        ax0.bar(x - width/2, fair_v1, width, label="V1 (Face)", color=COLORS, alpha=0.8, edgecolor='black')
        ax0.bar(x + width/2, fair_v2, width, label="V2 (Unique)", color=COLORS, alpha=0.6, hatch='//', edgecolor='black')
        ax0.set_title("Fair Win Rates", fontsize=14)
        ax0.set_xticks(x)
        ax0.set_xticklabels(pet_names)
        ax0.set_ylim(0, 1)
        ax0.grid(axis='y', alpha=0.3)
        for i, (v1, v2) in enumerate(zip(fair_v1, fair_v2)):
            ax0.text(i - width/2, v1 + 0.01, f"{v1:.0%}", ha='center', va='bottom')
            ax0.text(i + width/2, v2 + 0.01, f"{v2:.0%}", ha='center', va='bottom')
        ax0.legend()

        scenario_info = [
            ("Punish Top", f"Punish Top: {top_name}", top_i),
            ("Boost 2nd", f"Boost 2nd: {second_name}", second_i),
            ("Boost 3rd", f"Boost 3rd: {third_name}", third_i),
        ]

        master_handles, master_labels = [], []
        for plot_idx, (scenario_key, title, target_idx) in enumerate(scenario_info, 1):
            ax = axes[plot_idx]
            bias_vals = sorted(results[scenario_key]["V1"].keys())
            for pet_i in range(4):
                v1 = [results[scenario_key]["V1"][b][pet_i] for b in bias_vals]
                v2 = [results[scenario_key]["V2"][b][pet_i] for b in bias_vals]
                line_v1, = ax.plot(bias_vals, v1, 'o-', color=COLORS[pet_i], label=f"{pet_names[pet_i]} (V1)")
                line_v2, = ax.plot(bias_vals, v2, 's--', color=COLORS[pet_i], label=f"{pet_names[pet_i]} (V2)")
                if plot_idx == 1:
                    master_handles.extend([line_v1, line_v2])
                    master_labels.extend([f"{pet_names[pet_i]} (V1)", f"{pet_names[pet_i]} (V2)"])
            ax.axvline(0, color='gray', ls=':')
            ax.set_title(title)
            ax.set_xlabel("Bias (%)")
            ax.set_ylabel("Win Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if plot_idx == 1: ax.invert_xaxis()
            y0 = results[scenario_key]["V1"][0][target_idx]
            ax.plot(0, y0, 'ko', markersize=6, markeredgecolor='white')

        fig.legend(master_handles, master_labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        st.pyplot(fig)

        # Export
        export_data = {"Fair_V1": fair_v1, "Fair_V2": fair_v2}
        for sc in ["Punish Top", "Boost 2nd", "Boost 3rd"]:
            for m in ["V1", "V2"]:
                for b, r in results[sc][m].items():
                    export_data[f"{sc}_{m}_bias{b:+}"] = r
        df = pd.DataFrame(export_data, index=pet_names)
        st.download_button("📥 Download CSV", df.to_csv(), "pet_racing_bias_analysis.csv", "text/csv")

if __name__ == "__main__":
    main()