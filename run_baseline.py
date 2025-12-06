import numpy as np
import pandas as pd
from random_baseline import RandomBaseline
from CoursePlan import CoursePlan
from game import Game 
from forward import A, State, BeliefState 

def display_quarter(game: Game, quarter): 
    """
    Prints out a quarter's contents: each course's name and unit count. Code copied from run.py for forward search, for formatting consistency.
    """
    for course_id in quarter:
        course_info = game.courses.loc[course_id]
        subjects = course_info['subject_codes']
        units = course_info['units']
        print(f"  - Course {course_id}: {subjects} ({units} units)")


def run_baseline_loop(game: CoursePlan, uniform_belief: np.ndarray, true_state_idx: int) -> tuple:
    """
    Run random baseline policy in a loop until 12 quarters have been planned. Code copied from run.py for forward search, for formatting consistency.
    """
    random = RandomBaseline(game, uniform_belief)
   
    #Tracking for pre-req checking and reward computation 
    known_state = ()
    belief_state = BeliefState(belief=uniform_belief, known_state=known_state)
    true_state = State(uncertain=true_state_idx, known=known_state)
    planned_quarters = []
    total_reward = 0.0

    print("=" * 80)
    print("COURSE PLANNING WITH RANDOM POLICY AS BASELINE")
    print("=" * 80)
    print(f"Number of subjects: {len(game.subjects)}")
    print(f"True student interest: {game.subjects[true_state_idx]}")
    print()

    quarter_num = 0
    while quarter_num < 12:
        quarter_num += 1
        print(f"\n{'=' * 80}")
        print(f"QUARTER {quarter_num}")
        print(f"{'=' * 80}")

        # Get available actions
        valid_actions = game.actions(belief_state)
        print(f"\nNumber of valid quarter options: {len(valid_actions)}")
        if not valid_actions:
            print("\nNo valid actions available. Planning complete!")
            break

        action = random.run(belief_state)
        if action is None:
            print("No action found. Stopping.")
            break

        # Display selected quarter
        print(f"\nSelected quarter:")
        display_quarter(game, action)
        # for course_id in action:
        #     course_info = game.courses.loc[course_id]
        #     subjects = course_info['subject_codes']
        #     units = course_info['units']
        #     print(f"  - Course {course_id}: {subjects} ({units} units)")
        planned_quarters.append(action)

        # Execute action in environment using step() 
        true_state, observation, reward = game.step(true_state, action) #only using reward generated here
        total_reward += reward
        print(f"Reward: {reward:.4f}")
        print(f"Cumulative reward: {total_reward:.4f}")

        # belief_state = belief_state.update(true_state.known)

        # Display cumulative units
        total_units = sum(
            game.courses.loc[course_id]['units']
            for quarter in planned_quarters
            for course_id in quarter
        )
        print(f"\nCumulative units: {total_units}")
    
    #Display full schedule
    print("\n" + "=" * 80)
    print("PLANNING COMPLETE")
    print("=" * 80)
    print(f"\nTotal quarters: {len(planned_quarters)}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"\nFull course schedule:")
    for i, quarter in enumerate(planned_quarters, 1):
        print(f"\nQuarter {i}:")
        display_quarter(game, quarter)
        # for course_id in quarter:
        #     course_info = game.courses.loc[course_id]
        #     subjects = course_info['subject_codes']
        #     units = course_info['units']
        #     print(f"  - Course {course_id}: {subjects} ({units} units)")
    
    total_units = sum(
        game.courses.loc[course_id]['units']
        for quarter in planned_quarters
        for course_id in quarter
    )
    print(f"\nTotal units completed: {total_units}")

    return planned_quarters, total_reward


def main():
    """
    Main function to run the random policy demo. Code copied/modified from run.py for forward search, for setup and formatting consistency
    """
    try: # Load course data
        courses_df = pd.read_csv('data/courses_processed.csv')
        # drop embedding column
        if 'embedding' in courses_df.columns:
            courses_df = courses_df.drop(columns=['embedding'])
        # rename umap_3d to embedding
        if 'umap_3d' in courses_df.columns:
            courses_df = courses_df.rename(columns={'umap_3d': 'embedding'})
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

    # Set true student interest (for simulation), can change this to test different scenarios
    if 'CS' in game.subjects:
        true_state_idx = game.subjects.index('CS')
    else:
        true_state_idx = 0  # Default to first subject

    # Set uniform belief over subjects, which will NOT be updated
    n_subjects = len(game.subjects)
    uniform_belief = np.ones(n_subjects) / n_subjects

    planned_quarters, total_reward = run_baseline_loop(
        game=game,
        uniform_belief=uniform_belief,
        true_state_idx=true_state_idx
    )

    print(f"\n\nSimulation complete!")
    print(f"Planned {len(planned_quarters)} quarters")
    print(f"Total reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()