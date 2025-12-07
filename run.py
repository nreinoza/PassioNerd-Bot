import numpy as np
import polars as pl
from forward import BeliefState, ForwardSearch, State
from CoursePlan import CoursePlan
import time


def run_forward_search_loop(
    game: CoursePlan,
    initial_belief: np.ndarray,
    true_state_idx: int,
):
    """
    Run forward search loop to plan course schedule until no actions available.
    
    Args:
        game: CoursePlan instance
        initial_belief: Initial probability distribution over subjects
        true_state_idx: True student interest (for simulation)
    
    Returns:
        Tuple of (planned_quarters, total_reward)
    """
    # Initialize forward search
    discount = 1.0
    search = ForwardSearch(game, discount=discount, value_fn=ForwardSearch.units_bonus_value_fn(game))
    
    # Initialize belief state with empty history
    known_state = ()  # Empty tuple of quarters
    belief_state = BeliefState(belief=initial_belief, known_state=known_state)
    
    # Initialize true state for simulation
    true_state = State(uncertain=true_state_idx, known=known_state)
    
    # Track planned quarters and rewards
    planned_quarters = []
    total_reward = 0.0
    
    print("=" * 80)
    print("COURSE PLANNING WITH FORWARD SEARCH")
    print("=" * 80)
    print(f"Discount factor: {discount}")
    print(f"Number of subjects: {len(game.subjects)}")
    print(f"True student interest: {game.subjects[true_state_idx]}")
    print()

    search_depth = 1
    
    # Main planning loop - run until no actions available
    quarter_num = 0
    while True:
        quarter_num += 1
        print(f"\n{'=' * 80}")
        print(f"QUARTER {quarter_num}")
        print(f"{'=' * 80}")
        
        # Display current belief
        print("\nCurrent belief over subjects:")
        belief_df = pl.DataFrame({
            'Subject': game.subjects,
            'Probability': belief_state.belief
        }).sort('Probability', descending=True)
        print(belief_df)
        
        # Perform forward search
        if quarter_num == 2:
            search_depth += 1
        print(f"\nPerforming forward search (depth={search_depth})...")
        best_action, best_value, all_action_values = search.search(belief_state, depth=search_depth)
        
        if best_action is None:
            print("No action found. Stopping.")
            break
        
        # Display top actions considered
        print(f"\nTop 5 quarter options considered:")
        for rank, (action, q_value) in enumerate(all_action_values[:5], 1):
            course_list = []
            total_units = 0
            for course_id in action:
                subjects = game.course_metadata[course_id]['subject_codes']
                units = game.course_metadata[course_id]['units']
                course_list.append(f"{course_id} ({subjects}, {units}u)")
                total_units += units
            courses_str = ", ".join(course_list)
            marker = "★" if rank == 1 else " "
            print(f"  {marker} {rank}. Q={q_value:.4f} | {total_units}u | {courses_str}")
        
        # Display selected quarter
        print(f"\nSelected quarter (Q-value: {best_value:.4f}):")
        for course_id in best_action:
            subjects = game.course_metadata[course_id]['subject_codes']
            units = game.course_metadata[course_id]['units']
            print(f"  - Course {course_id}: {subjects} ({units} units)")
        
        planned_quarters.append(best_action)
        
        # Execute action in environment using step()
        true_state, observation, reward = game.step(true_state, best_action)
        total_reward += reward
        
        observed_subject = game.subjects[observation]
        print(f"\nObservation: {observed_subject}")
        print(f"Reward: {reward:.4f}")
        print(f"Cumulative reward: {total_reward:.4f}")
        
        # Update belief based on observation
        obs_probs_matrix = game.observation_probs(best_action)
        likelihood = obs_probs_matrix[observation, :]
        belief_state = belief_state.update(observation, likelihood, true_state.known)
        
        print(f"\nUpdated belief (top 5 subjects):")
        updated_belief_df = pl.DataFrame({
            'Subject': game.subjects,
            'Probability': belief_state.belief
        }).sort('Probability', descending=True).head(5)
        print(updated_belief_df)
        
        # Display cumulative units
        total_units = sum(
            game.course_metadata[course_id]['units']
            for quarter in planned_quarters
            for course_id in quarter
        )
        print(f"\nCumulative units: {total_units}")
    
    print("\n" + "=" * 80)
    print("PLANNING COMPLETE")
    print("=" * 80)
    print(f"\nTotal quarters: {len(planned_quarters)}")
    print(f"Total reward: {total_reward:.4f}")
    
    # Display full schedule
    print(f"\nFull course schedule:")
    for i, quarter in enumerate(planned_quarters, 1):
        print(f"\nQuarter {i}:")
        for course_id in quarter:
            subjects = game.course_metadata[course_id]['subject_codes']
            units = game.course_metadata[course_id]['units']
            print(f"  - Course {course_id}: {subjects} ({units} units)")
    
    total_units = sum(
        game.course_metadata[course_id]['units']
        for quarter in planned_quarters
        for course_id in quarter
    )
    print(f"\nTotal units completed: {total_units}")
    
    return planned_quarters, total_reward


def main():
    """
    Main function to run the forward search demo.
    """
    # Load course data
    try:
        courses_df = pl.read_csv('data/courses_processed.csv')
        # drop embedding column
        if 'embedding' in courses_df.columns:
            courses_df = courses_df.drop('embedding')
        # rename umap_3d to embedding
        if 'umap_3d' in courses_df.columns:
            courses_df = courses_df.rename({'umap_3d': 'embedding'})
    except FileNotFoundError:
        print("Error: courses.csv not found.")
        print("Please provide a CSV file with columns: id, subject_codes, units, embedding")
        return
    
    # Initialize game
    print("Initializing course planning game...")
    game = CoursePlan(courses_df)
    print(f"Loaded {len(game.courses)} courses")
    print(f"Found {len(game.subjects)} subjects: {', '.join(game.subjects)}")
    print()
    
    # Set initial uniform belief over all subjects
    n_subjects = len(game.subjects)
    initial_belief = np.ones(n_subjects) / n_subjects
    
    # Set true student interest (for simulation)
    # You can change this to test different scenarios
    if 'BIO' in game.subjects:
        true_state_idx = game.subjects.index('BIO')
    else:
        true_state_idx = 0  # Default to first subject
    
    # Alternatively, you can set a non-uniform initial belief
    # For example, if student has some prior knowledge about their interests:
    # initial_belief = np.ones(n_subjects) / n_subjects
    # cs_idx = game.subjects.index('CS') if 'CS' in game.subjects else 0
    # initial_belief[cs_idx] = 0.3
    # initial_belief = initial_belief / initial_belief.sum()  # Renormalize
    
    # Run forward search
    start = time.perf_counter()
    planned_quarters, total_reward = run_forward_search_loop(
        game=game,
        initial_belief=initial_belief,
        true_state_idx=true_state_idx,
    )
    end = time.perf_counter()
    
    print(f"\n\nSimulation complete!")
    print(f"Planned {len(planned_quarters)} quarters")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()