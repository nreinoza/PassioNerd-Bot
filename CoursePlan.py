import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
import re
from collections import defaultdict

# Import from the POMDP framework
from forward import Game, State, BeliefState


# Type alias for KnownState: tuple of quarters, where each quarter is a tuple of course IDs
KnownState = Tuple[Tuple[int, ...], ...]


class CoursePlan(Game):
    """
    POMDP formulation of course planning.
    
    Uncertain state: Index into list of subjects (representing student's interest/preference)
    Known state: History of courses taken, organized by quarters
    Actions: Enrolling in specific courses
    Observations: After taking a course, observe which subject area it belongs to
    """
    
    def __init__(self, courses: pd.DataFrame):
        """
        Initialize the course planning game.
        
        Args:
            courses: DataFrame with columns [id, subject_codes, units, embedding]
        """
        # Store original dataframe for metadata (units, embeddings)
        self.courses = courses.copy()
        
        # Ensure embeddings are numpy arrays
        self.courses['embedding'] = self.courses['embedding'].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
        )
        
        # Set id as index for O(1) lookup performance
        self.courses.set_index('id', inplace=True)
        
        # Create indexed dataframe for prerequisite logic
        self.course_codes = self._create_course_codes(courses)
        
        # Extract all unique subjects (sorted for consistent indexing)
        self.subjects = sorted(self.course_codes['subject'].unique().tolist())
        
        # Compute mean embeddings for each subject
        self.subject_embeddings = self._compute_subject_embeddings()
    
    def _create_course_codes(self, courses: pd.DataFrame) -> pd.DataFrame:
        """
        Create indexed dataframe with one row per (course_id, subject) pair.
        
        Returns:
            DataFrame with columns [id, subject, number, level]
            where level is 0 (intro), 1 (intermediate), or 2 (advanced)
        """
        rows = []
        
        for _, row in courses.iterrows():
            course_id = int(row['id'])
            subject_codes_str = row['subject_codes']
            
            # Parse each subject code (handles MS&E, AA, CS, etc.)
            for code in subject_codes_str.split(','):
                code = code.strip()
                # Match: subject (any non-digit chars) + space + number
                match = re.match(r'^([^\d]+)\s+(\d+)', code)
                if match:
                    subject = match.group(1).strip()
                    number = int(match.group(2))
                    
                    # Determine level: 0 (intro), 1 (intermediate), 2 (advanced)
                    if number < 110:
                        level = 0
                    elif number < 200:
                        level = 1
                    else:
                        level = 2
                    
                    rows.append({
                        'id': course_id,
                        'subject': subject,
                        'number': number,
                        'level': level
                    })
        
        return pd.DataFrame(rows)
    
    def _compute_subject_embeddings(self) -> np.ndarray:
        """
        Compute mean embedding for each subject.
        
        Returns:
            Array of shape (n_subjects, embedding_dim) where row i is mean embedding for subjects[i]
        """
        # Map subject to index
        subject_to_idx = {subj: i for i, subj in enumerate(self.subjects)}
        
        # Get embedding dimension
        embedding_dim = len(self.courses.iloc[0]['embedding'])
        subject_embeddings = np.zeros((len(self.subjects), embedding_dim))
        subject_counts = np.zeros(len(self.subjects))
        
        # Accumulate embeddings for each subject
        for course_idx, row in self.courses.iterrows():
            course_id = int(course_idx)  # course_idx is now the course id since we set it as index
            embedding = row['embedding']
            
            # Get all subjects for this course from course_codes
            course_subjects = self.course_codes[self.course_codes['id'] == course_id]['subject'].unique()
            
            for subject in course_subjects:
                idx = subject_to_idx[subject]
                subject_embeddings[idx] += embedding
                subject_counts[idx] += 1
        
        # Compute means (avoid division by zero)
        for i in range(len(self.subjects)):
            if subject_counts[i] > 0:
                subject_embeddings[i] /= subject_counts[i]
        
        return subject_embeddings
    
    def _count_courses_by_subject(self, known_state: KnownState) -> pd.DataFrame:
        """
        Count courses taken by subject and level.
        
        Args:
            known_state: History of quarters with course IDs
        
        Returns:
            DataFrame with columns [subject, level, count]
            where level is 0 (intro), 1 (intermediate), or 2 (advanced)
        """
        # Flatten all course IDs from all quarters
        all_course_ids = []
        for quarter in known_state:
            all_course_ids.extend(quarter)
        
        if not all_course_ids:
            # Return empty dataframe with correct schema
            return pd.DataFrame(columns=['subject', 'level', 'count'])
        
        # Filter course_codes to taken courses
        taken = self.course_codes[self.course_codes['id'].isin(all_course_ids)]
        
        # Count by subject and level
        counts = taken.groupby(['subject', 'level']).size().reset_index(name='count')
        
        return counts
    
    def actions(self, belief_state: BeliefState) -> List[Tuple[int, ...]]:
        """
        Get list of valid quarters (groups of 4 courses) that can be taken.
        
        Eligibility criteria:
        - Intro courses (level 0): always available
        - Intermediate courses (level 1): need 3+ intro courses in that subject
        - Advanced courses (level 2): need 2+ intermediate courses in that subject
        
        Args:
            belief_state: Current belief state (contains belief over interests and course history)
        
        Returns:
            List of quarters, where each quarter is a tuple of 4 course IDs
        """
        # Extract known state from belief state
        known_state = belief_state.known_state

        if len(known_state) == 3:
            # Max quarters reached
            return []
        
        # Get courses already taken
        taken_course_ids = set()
        for quarter in known_state:
            taken_course_ids.update(quarter)
        
        # Get prerequisite counts
        counts = self._count_courses_by_subject(known_state)
        
        # Create lookup dict: (subject, level) -> count
        prereq_counts = {}
        for _, row in counts.iterrows():
            key = (row['subject'], row['level'])
            prereq_counts[key] = row['count']
        
        # Find eligible courses
        eligible_by_subject = {subject: set() for subject in self.subjects}
        
        # Intro (level 0) always eligible
        for _, row in self.course_codes[self.course_codes['level'] == 0].iterrows():
            if row['id'] not in taken_course_ids:
                eligible_by_subject[row['subject']].add(row['id'])
        
        # Intermediate + Advanced based on prerequisites
        for subject in self.subjects:
            intro_count = prereq_counts.get((subject, 0), 0)
            intermediate_count = prereq_counts.get((subject, 1), 0)
            
            # Intermediate requires 3+ intro
            if intro_count >= 3:
                df_inter = self.course_codes[
                    (self.course_codes['subject'] == subject) & 
                    (self.course_codes['level'] == 1)
                ]
                for _, row in df_inter.iterrows():
                    if row['id'] not in taken_course_ids:
                        eligible_by_subject[subject].add(row['id'])
                
                # Advanced requires 2+ intermediate
                if intermediate_count >= 2:
                    df_adv = self.course_codes[
                        (self.course_codes['subject'] == subject) & 
                        (self.course_codes['level'] == 2)
                    ]
                    for _, row in df_adv.iterrows():
                        if row['id'] not in taken_course_ids:
                            eligible_by_subject[subject].add(row['id'])

        # Belief distribution over subjects
        belief = np.array(belief_state.belief)

        # Create sampling pool: subjects with at least 1 eligible course
        valid_subjects = [s for s in self.subjects if len(eligible_by_subject[s]) > 0]
        if len(valid_subjects) == 0:
            return []
        
        # Adjust belief to zero out subjects with no eligible courses
        adjusted_belief = np.array([
            belief[i] if self.subjects[i] in valid_subjects else 0.0 for i in range(len(self.subjects))
        ])  
        adjusted_belief = adjusted_belief / adjusted_belief.sum()
        
        # Naive sampling: generate random quarters of 4 courses
        # Sample up to 2 random quarters (or fewer if not enough eligible courses)
        quarters = []
        n_samples = min(2, sum(len(v) for v in eligible_by_subject.values()) // 4)
        
        for _ in range(n_samples):
            quarter_courses = []
            
            for _ in range(4):
                # Sample a subject
                subject_idx = np.random.choice(len(self.subjects), p=adjusted_belief)
                subject = self.subjects[subject_idx]
                
                # Get eligible non-used courses for subject
                available = eligible_by_subject[subject] - set(quarter_courses)
                if not available:
                    break  # cannot complete this quarter, skip it
                
                # Uniformly choose an eligible course within the subject
                # Convert set to list for np.random.choice
                course = np.random.choice(list(available))
                quarter_courses.append(course)
            
            # Only accept full quarters
            if len(quarter_courses) == 4:
                quarters.append(tuple(sorted(quarter_courses)))
        
        return quarters
    
    def reward(self, state: State, action: Tuple[int, ...]) -> float:
        """
        Get reward for taking a quarter of courses (action) from a state.
        
        TODO: Define reward function based on:
        - Alignment between student's uncertain interest (state.uncertain) and course subject
        - Course quality/value
        - Progress towards degree requirements
        
        Args:
            state: Complete state (uncertain interest + known history)
            action: Quarter (tuple of course IDs) to take
        
        Returns:
            Reward value
        """
        total = 0.0
        course_ids = action
               

        # for each course_id
        for course_id in course_ids:
            # Get the course data using index-based lookup (O(1))
            course_embedding = self.courses.loc[course_id, 'embedding']
            units = self.courses.loc[course_id, 'units']

            # 1. compute euclidean distance between self.true_pref and this courses' embedding
            distance = np.linalg.norm(self.subject_embeddings[state.uncertain] - course_embedding)
            # Negative distance as reward
            total -= distance * units
        
        return total
    
    def observation_probs(self, action: Tuple[int, ...]) -> np.ndarray:
        """
        Get observation probability matrix for taking a quarter of courses.
        
        After taking a quarter, we observe which subjects the courses belong to.
        
        Args:
            action: Quarter (tuple of course IDs) being taken
        
        Returns:
            Matrix of shape (n_observations, n_uncertain_states) where entry [i, j] is
            P(observation=i | uncertain_state=j, action)
        """
        n_subjects = len(self.subjects)
        sigma = 1.0  # Observation noise parameter (can be tuned)
        
        # Get embeddings for all courses in the quarter
        course_embeddings = []
        for course_id in action:
            course_embeddings.append(self.courses.loc[course_id, 'embedding'])
        
        # Initialize probability matrix: P(obs=i | state=j, action)
        # Shape: (n_observations, n_uncertain_states)
        obs_probs = np.zeros((n_subjects, n_subjects))
        
        # For each true state (student's true interest)
        for state_idx, true_subject in enumerate(self.subjects):
            true_embedding = self.subject_embeddings[state_idx]
            
            # Compute distances from true interest to each course
            true_distances = []
            for course_emb in course_embeddings:
                dist = np.linalg.norm(true_embedding - course_emb)
                true_distances.append(dist)
            
            # For each possible observation
            for obs_idx, obs_subject in enumerate(self.subjects):
                obs_embedding = self.subject_embeddings[obs_idx]
                
                # Compute distances from observed subject to each course
                obs_distances = []
                for course_emb in course_embeddings:
                    dist = np.linalg.norm(obs_embedding - course_emb)
                    obs_distances.append(dist)
                
                # Compute log probability (for numerical stability)
                # P(o | s, a) ∝ exp(-1/(2σ²) * Σ(d(v_o, e_i) - d(v_s, e_i))²)
                sum_squared_diff = 0.0
                for i in range(len(course_embeddings)):
                    diff = obs_distances[i] - true_distances[i]
                    sum_squared_diff += diff ** 2
                
                log_prob = -sum_squared_diff / (2 * sigma ** 2)
                obs_probs[obs_idx, state_idx] = log_prob
        
        # Convert from log probabilities to probabilities and normalize
        # For numerical stability, subtract max before exponentiating
        for state_idx in range(n_subjects):
            log_probs = obs_probs[:, state_idx]
            max_log_prob = np.max(log_probs)
            probs = np.exp(log_probs - max_log_prob)
            # Normalize to sum to 1
            obs_probs[:, state_idx] = probs / np.sum(probs)
        
        return obs_probs
    
    def transition(self, known_state: KnownState, action: Tuple[int, ...]) -> KnownState:
        """
        Update known state after taking a quarter of courses.
        
        Adds the quarter to the known_state history.
        
        Args:
            known_state: Current history of quarters with course IDs
            action: Quarter (tuple of course IDs) being taken
        
        Returns:
            New known state with quarter added
        """
        # Add the quarter to the history
        return known_state + (action,)