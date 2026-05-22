import numpy as np




def check_trial_alignment(act: np.ndarray, handvel: np.ndarray) -> bool:
    """
    Verify that act and handvel have:
      - the same number of sessions
      - the same number of trials in each session

    Parameters
    ----------
    act     : object array (n_sessions,), each element (n_trials, T, N)
    handvel : object array (n_sessions,), each element (n_trials, 2, T)

    Returns
    -------
    True if all checks pass, raises AssertionError otherwise.
    """
    assert len(act) == len(handvel), (
        f"Session count mismatch: act has {len(act)} sessions, "
        f"handvel has {len(handvel)} sessions."
    )

    # print(f"{'Session':>8} | {'act trials':>12} | {'handvel trials':>14} | {'Match':>6}")
    # print("-" * 50)

    all_match = True
    for i, (a, h) in enumerate(zip(act, handvel)):
        a_trials = np.array(a).shape[0]
        h_trials = np.array(h).shape[0]
        match    = a_trials == h_trials
        all_match = all_match and match
        status   = "OK" if match else "MISMATCH"
        # print(f"{i:>8} | {a_trials:>12} | {h_trials:>14} | {status:>6}")

    # print()
    assert all_match, "Trial count mismatch detected in one or more sessions."
    # print("OK: All sessions have matching trial counts.")
    return True



def concatenate_sessions(act: np.ndarray, handvel: np.ndarray,
                         fold: int =0, test_ratio: float = 0.2, seed: int = 42):
    """
    Concatenate all sessions and split into train/test by randomly
    sampling ~test_ratio trials from EACH session independently.
 
    - act     sessions: (n_trials, T, N)  → keep as-is
    - handvel sessions: (n_trials, 2, T)  → transpose to (n_trials, T, 2)
 
    Parameters
    ----------
    act        : object array (n_sessions,), each (n_trials, T, N)
    handvel    : object array (n_sessions,), each (n_trials, 2, T)
    test_ratio : float, fraction of trials per session for test (default 0.2)
    seed       : int, random seed for reproducibility (default 42)
 
    Returns
    -------
    neural_train, neural_test     : np.ndarray  (n_trials, T, N)
    behavior_train, behavior_test : np.ndarray  (n_trials, T, 2)
    """
    rng = np.random.default_rng(seed)
 
    neural_train_list,   neural_test_list   = [], []
    behavior_train_list, behavior_test_list = [], []
 
    # print(f"  {'Sess':>4} | {'Total':>6} | {'Train':>6} | {'Test':>5} | {'Test%':>6}")
    # print(f"  {'-'*38}")
 
    for i, (a, h) in enumerate(zip(act, handvel)):
        a = np.array(a)                      # (n_trials, T, N)
        h = np.array(h).transpose(0, 2, 1)  # (n_trials, T, 2)
 
        n_trials  = a.shape[0]
        n_test    = max(1, round(n_trials * test_ratio))
        n_train   = n_trials - n_test

        fold=fold
        idx       = rng.permutation(n_trials)
        start = int(fold * n_test)
        end = int(start + n_test)
        test_idx  = idx[start:end]

        train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)

        train_idx = np.sort(train_idx)
        test_idx  = np.sort(test_idx)
 
        neural_train_list.append(a[train_idx])
        neural_test_list.append(a[test_idx])
        behavior_train_list.append(h[train_idx])
        behavior_test_list.append(h[test_idx])
 
        # print(f"  {i:>4} | {n_trials:>6} | {n_train:>6} | {n_test:>5} | {n_test/n_trials:.1%}")
 
    neural_train   = np.concatenate(neural_train_list,   axis=0)
    neural_test    = np.concatenate(neural_test_list,    axis=0)
    behavior_train = np.concatenate(behavior_train_list, axis=0)
    behavior_test  = np.concatenate(behavior_test_list,  axis=0)
 
    total        = neural_train.shape[0] + neural_test.shape[0]
    # print(f"  {'-'*38}")
    # print(f"  {'ALL':>4} | {total:>6} | {neural_train.shape[0]:>6} | "
    #       f"{neural_test.shape[0]:>5} | {neural_test.shape[0]/total:.1%}")
    # print(f"\nPer-session random split (seed={seed}):")
    # print(f"  neural_train {neural_train.shape} / neural_test {neural_test.shape}")
    # print(f"  behavior_train {behavior_train.shape} / behavior_test {behavior_test.shape}\n")
 
    return neural_train, neural_test, behavior_train, behavior_test


# ─────────────────────────────────────────────────────────────
# (3) Generate labels
# ─────────────────────────────────────────────────────────────
def generate_labels(behavior: np.ndarray,
                    split_point: int = 100) -> np.ndarray:
    """
    Generate a label array matching the behavior array's trial/time dimensions.

    Label definition:
        time points [0,   split_point)  → 0  (e.g. pre-movement)
        time points [split_point, T)    → 1  (e.g. movement)

    Parameters
    ----------
    behavior    : np.ndarray  (total_trials, T, D)
    split_point : int         time index where label switches from 0 to 1

    Returns
    -------
    labels : np.ndarray  (total_trials, T, 1)  dtype int
    """
    total_trials, T, _ = behavior.shape
    assert T >= split_point, (
        f"split_point ({split_point}) must be ≤ number of time points ({T})."
    )

    label_template = np.zeros(T, dtype=int)
    label_template[split_point:] = 1                          # (T,)

    labels = np.tile(label_template, (total_trials, 1))       # (total_trials, T)
    labels = labels[:, :, np.newaxis]                         # (total_trials, T, 1)

    # print(f"Generated labels shape: {labels.shape}  (trials, time, 1)")
    # print(f"  Label 0: time points [0, {split_point})   →  pre-movement")
    # print(f"  Label 1: time points [{split_point}, {T}) →  movement\n")
    return labels


def process(act_path: str, handvel_path: str,hand_path: str,force_path: str, fold:int =0, split_point: int = 100):
    """
    Full pipeline: load → validate → concatenate + split → label.
 
    Parameters
    ----------
    act_path     : path to act_save_single_200.npy
    handvel_path : path to handvel_save_single_200.npy
    split_point  : time index where label switches from 0 to 1 (default 100)
 
    Returns
    -------
    neural_train, neural_test     : np.ndarray  (n_trials, T, N)
    behavior_train, behavior_test : np.ndarray  (n_trials, T, 2)
    labels_train, labels_test     : np.ndarray  (n_trials, T, 1)
    """
    # Load
    act     = np.load(act_path,     allow_pickle=True)
    handvel = np.load(handvel_path, allow_pickle=True)
    hand    = np.load(hand_path,     allow_pickle=True)
    force = np.load(force_path, allow_pickle=True)


    # print(f"Loaded act:     {len(act)} sessions")
    # print(f"Loaded handvel: {len(handvel)} sessions")
    # print(f"Loaded hand: {len(hand)} sessions")
    # print(f"Loaded force: {len(force)} sessions")
 
    # (1) Check trial alignment
    check_trial_alignment(act, handvel)
    check_trial_alignment(act, hand)
    check_trial_alignment(act, force)
 
    # (2) Concatenate + split
    neural_train, neural_test, handvel_train, handvel_test = \
        concatenate_sessions(act, handvel, fold=fold,test_ratio=0.2, seed=42)
    _, _, hand_train, hand_test = \
        concatenate_sessions(act, hand, fold=fold,test_ratio=0.2, seed=42)
    _, _, force_train, force_test = \
        concatenate_sessions(act, force, fold=fold,test_ratio=0.2, seed=42)
 
    # (3) Labels for train and test separately
    labels_train = generate_labels(handvel_train, split_point=split_point)
    labels_test  = generate_labels(handvel_test,  split_point=split_point)
 
    return neural_train, neural_test, handvel_train, handvel_test, hand_train, hand_test, force_train, force_test, labels_train, labels_test

def normalize_train_test(train, test, eps=1e-8):
    
    scale = max(np.abs(train).max(), np.abs(test).max())
    scale = max(scale, eps)  # avoid divide-by-zero
    
    return train / scale, test / scale, scale