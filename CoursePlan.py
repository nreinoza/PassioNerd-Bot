import numpy as np
import polars as pl
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

    MAX_QUARTERS = 12
    
    def __init__(self, courses: pl.DataFrame):
        """
        Initialize the course planning game.
        
        Args:
            courses: DataFrame with columns [id, subject_codes, units, embedding]
        """
        # Convert embeddings to numpy matrix (all at once)
        course_ids = courses['id'].to_list()
        embeddings_list = []
        units_list = []
        subject_codes_list = []
        
        for row in courses.iter_rows(named=True):
            embedding = row['embedding']
            if isinstance(embedding, str):
                embedding = np.array(eval(embedding))
            else:
                embedding = np.array(embedding)
            embeddings_list.append(embedding)
            units_list.append(row['units'])
            subject_codes_list.append(row['subject_codes'])
        
        # Vectorized storage
        self.course_ids = np.array(course_ids, dtype=np.int32)
        self.course_embeddings_matrix = np.vstack(embeddings_list)  # Shape: (n_courses, embedding_dim)
        self.course_units = np.array(units_list, dtype=np.float32)
        
        # Create mapping from course_id to index
        self.course_id_to_idx = {cid: idx for idx, cid in enumerate(course_ids)}
        
        # Store metadata separately
        self.course_metadata = {
            int(cid): {'units': units, 'subject_codes': sc}
            for cid, units, sc in zip(course_ids, units_list, subject_codes_list)
        }
        
        # Store DataFrame WITHOUT embeddings (for serialization)
        self.courses = courses.select(['id', 'subject_codes', 'units'])
        
        # Create indexed dataframe for prerequisite logic
        self.course_codes = self._create_course_codes(courses)
        
        # Extract all unique subjects (sorted for consistent indexing)
        self.subjects = sorted(self.course_codes.select('subject').unique().to_series().to_list())
        
        # Pre-compute subject to courses mapping
        self._precompute_subject_mappings()
        
        # Compute mean embeddings for each subject (vectorized)
        self.subject_embeddings = self._compute_subject_embeddings_vectorized()
    
    def _create_course_codes(self, courses: pl.DataFrame) -> pl.DataFrame:
        """
        Create indexed dataframe with one row per (course_id, subject) pair.
        
        Returns:
            DataFrame with columns [id, subject, number, level]
            where level is 0 (intro), 1 (intermediate), or 2 (advanced)
        """
        rows = []
        
        for row in courses.iter_rows(named=True):
            course_id = int(row['id'])
            subject_codes_str = row['subject_codes']
            
            # Parse each subject code
            for code in subject_codes_str.split(','):
                code = code.strip()
                match = re.match(r'^([^\d]+)\s+(\d+)', code)
                if match:
                    subject = match.group(1).strip()
                    number = int(match.group(2))
                    
                    # Vectorizable level computation
                    level = 0 if number < 110 else (1 if number < 200 else 2)
                    
                    rows.append({
                        'id': course_id,
                        'subject': subject,
                        'number': number,
                        'level': level
                    })
        
        return pl.DataFrame(rows)
    
    def _precompute_subject_mappings(self):
        """
        Pre-compute mappings from subject/level to course indices for faster lookup.
        """
        # Map: subject -> set of course indices (in self.course_ids)
        self.subject_to_course_indices = defaultdict(set)
        
        # Map: (subject, level) -> set of course indices
        self.subject_level_to_course_indices = defaultdict(set)
        
        for row in self.course_codes.iter_rows(named=True):
            course_id = row['id']
            if course_id in self.course_id_to_idx:
                idx = self.course_id_to_idx[course_id]
                subject = row['subject']
                level = row['level']
                
                self.subject_to_course_indices[subject].add(idx)
                self.subject_level_to_course_indices[(subject, level)].add(idx)
    
    def _compute_subject_embeddings_vectorized(self) -> np.ndarray:
        """
        Compute mean embedding for each subject using vectorized operations.
        
        Returns:
            Array of shape (n_subjects, embedding_dim)
        """
        embedding_dim = self.course_embeddings_matrix.shape[1]
        n_subjects = len(self.subjects)
        
        subject_embeddings = np.zeros((n_subjects, embedding_dim))
        
        # Group by subject and compute means
        for subj_idx, subject in enumerate(self.subjects):
            course_indices = list(self.subject_to_course_indices[subject])
            if course_indices:
                # Vectorized mean computation
                subject_embeddings[subj_idx] = self.course_embeddings_matrix[course_indices].mean(axis=0)
        
        return subject_embeddings
    
    def _count_courses_by_subject(self, known_state: KnownState) -> pl.DataFrame:
        """
        Count courses taken by subject and level.
        
        Args:
            known_state: History of quarters with course IDs
        
        Returns:
            DataFrame with columns [subject, level, count]
        """
        # Flatten all course IDs using comprehension (faster than nested loop)
        all_course_ids = [cid for quarter in known_state for cid in quarter]
        
        if not all_course_ids:
            return pl.DataFrame(schema={'subject': pl.Utf8, 'level': pl.Int64, 'count': pl.UInt32})
        
        # Vectorized filtering and grouping
        counts = (
            self.course_codes
            .filter(pl.col('id').is_in(all_course_ids))
            .group_by(['subject', 'level'])
            .agg(pl.len().alias('count'))
        )
        
        return counts
    
    def actions(self, belief_state: BeliefState) -> List[Tuple[int, ...]]:
        """
        Get list of valid quarters (groups of courses) that can be taken.
        """
        known_state = belief_state.known_state

        if len(known_state) == self.MAX_QUARTERS:
            return []
        
        quarters_taken = len(known_state)
        in_first_half = quarters_taken < self.MAX_QUARTERS / 2
        
        min_units = 9 if in_first_half else 13
        max_units = 15 if in_first_half else 19
        
        # Flatten taken courses (vectorized)
        taken_course_ids = set(cid for quarter in known_state for cid in quarter)
        
        # Get prerequisite counts
        counts = self._count_courses_by_subject(known_state)
        prereq_counts = {(row['subject'], row['level']): row['count'] 
                        for row in counts.iter_rows(named=True)}
        
        # Find eligible courses using pre-computed mappings
        eligible_by_subject = {subject: set() for subject in self.subjects}
        
        # Intro (level 0) always eligible
        for subject in self.subjects:
            intro_indices = self.subject_level_to_course_indices[(subject, 0)]
            eligible_by_subject[subject].update(
                self.course_ids[idx] for idx in intro_indices 
                if self.course_ids[idx] not in taken_course_ids
            )
        
        # Intermediate + Advanced based on prerequisites
        for subject in self.subjects:
            intro_count = prereq_counts.get((subject, 0), 0)
            intermediate_count = prereq_counts.get((subject, 1), 0)
            
            if intro_count >= 3:
                inter_indices = self.subject_level_to_course_indices[(subject, 1)]
                eligible_by_subject[subject].update(
                    self.course_ids[idx] for idx in inter_indices 
                    if self.course_ids[idx] not in taken_course_ids
                )
                
                if intermediate_count >= 3:
                    adv_indices = self.subject_level_to_course_indices[(subject, 2)]
                    eligible_by_subject[subject].update(
                        self.course_ids[idx] for idx in adv_indices 
                        if self.course_ids[idx] not in taken_course_ids
                    )

        # Belief distribution over subjects
        belief = np.array(belief_state.belief)

        # Vectorized belief adjustment
        valid_subjects = [s for s in self.subjects if len(eligible_by_subject[s]) > 0]
        if not valid_subjects:
            return []
        
        # Create mask and apply in one operation
        subject_mask = np.array([self.subjects[i] in valid_subjects for i in range(len(self.subjects))])
        adjusted_belief = belief * subject_mask
        adjusted_belief = adjusted_belief / adjusted_belief.sum()
        
        # Build quarters
        quarters = []
        n_samples = self.num_actions(quarters_taken)
        
        while len(quarters) < n_samples:
            quarter_courses = []
            total_units = 0
            
            while True:
                subject_idx = np.random.choice(len(self.subjects), p=adjusted_belief)
                subject = self.subjects[subject_idx]
                
                available = eligible_by_subject[subject] - set(quarter_courses)
                if not available:
                    continue
                
                course = np.random.choice(list(available))
                course_units = self.course_metadata[course]['units']
                
                if total_units + course_units > max_units:
                    continue
                
                quarter_courses.append(course)
                total_units += course_units
                
                if total_units > max_units - 3:
                    break
                
                if total_units >= min_units and np.random.random() < 0.45:
                    break
            
            if total_units >= min_units:
                quarters.append(tuple(sorted(quarter_courses)))
        
        return quarters
    
    def num_actions(self, quarters_taken) -> int:
        """Calculate number of actions to sample (quarters)."""
        actions_per_q = {0: 10, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 
                        6: 4, 7: 4, 8: 4, 9: 4, 10: 6, 11: 12}
        return actions_per_q[quarters_taken]
    
    def reward(self, state: State, action: Tuple[int, ...]) -> float:
        """
        Get reward for taking a quarter of courses (vectorized).
        """
        course_ids = np.array(action, dtype=np.int32)
        
        # Get indices for all courses at once
        course_indices = np.array([self.course_id_to_idx[cid] for cid in course_ids])
        
        # Vectorized distance computation
        student_embedding = self.subject_embeddings[state.uncertain]
        course_embeddings = self.course_embeddings_matrix[course_indices]
        
        # Compute all distances at once: ||student - course||
        distances = np.linalg.norm(course_embeddings - student_embedding, axis=1)
        
        # Get units for all courses at once
        course_units = self.course_units[course_indices]
        
        # Vectorized enjoyment calculation
        enjoyment = -np.sum(distances * course_units)
        
        # Track subjects in this quarter
        subjects_in_quarter = set()
        for course_id in course_ids:
            course_subjects = (
                self.course_codes
                .filter(pl.col('id') == course_id)
                .select('subject')
                .unique()
                .to_series()
                .to_list()
            )
            subjects_in_quarter.update(course_subjects)

        # Scale enjoyment
        num_courses = len(course_ids)
        # Extend known state with current action for calculations
        extended_known = state.known + (action,)
        quarters_taken = len(extended_known)
        in_first_half = quarters_taken < self.MAX_QUARTERS / 2
        
        scaling_factor = 1 + ((num_courses + (1 if in_first_half else 0)) / 10)
        total = enjoyment * scaling_factor
        
        # Penalty for single-subject quarters
        if len(subjects_in_quarter) == 1:
            total -= 40

        # Bonus for completing a major
        if quarters_taken == self.MAX_QUARTERS:
            taken_course_ids = set(cid for quarter in extended_known for cid in quarter)

            # Vectorized units calculation
            taken_indices = np.array([self.course_id_to_idx[cid] for cid in taken_course_ids])
            num_units_taken = np.sum(self.course_units[taken_indices])
            
            if num_units_taken >= self.MAX_QUARTERS * 15 - (4 * self.MAX_QUARTERS / 2):
                taken_course_by_subject = self._count_courses_by_subject(extended_known)

                num_advanced_by_subject = defaultdict(int)
                for row in taken_course_by_subject.iter_rows(named=True):
                    if row['level'] == 2:
                        num_advanced_by_subject[row['subject']] += row['count']
                
                for subject, count in num_advanced_by_subject.items():
                    if count >= 10 and num_advanced_by_subject[subject] >= 2:
                        total += 500
                        break
        
        return total
    
    def observation_probs(self, action: Tuple[int, ...]) -> np.ndarray:
        """
        Get observation probability matrix (fully vectorized).
        """
        n_subjects = len(self.subjects)
        sigma = 1.55
        
        # Get course indices and embeddings (vectorized)
        course_indices = np.array([self.course_id_to_idx[cid] for cid in action])
        course_embeddings = self.course_embeddings_matrix[course_indices]  # Shape: (n_courses, embed_dim)
        
        # Compute all distances at once
        # subject_embeddings: (n_subjects, embed_dim)
        # course_embeddings: (n_courses, embed_dim)
        
        # Expand dimensions for broadcasting
        # subject_embeddings_expanded: (n_subjects, 1, embed_dim)
        # course_embeddings_expanded: (1, n_courses, embed_dim)
        subject_expanded = self.subject_embeddings[:, np.newaxis, :]
        course_expanded = course_embeddings[np.newaxis, :, :]
        
        # Compute all pairwise distances: (n_subjects, n_courses)
        all_distances = np.linalg.norm(subject_expanded - course_expanded, axis=2)
        
        # For each true state, compute observation probabilities
        obs_probs = np.zeros((n_subjects, n_subjects))
        
        for state_idx in range(n_subjects):
            true_distances = all_distances[state_idx]  # Shape: (n_courses,)
            
            # Vectorized computation of squared differences
            # obs_distances: (n_subjects, n_courses)
            # true_distances: (n_courses,)
            diff_matrix = all_distances - true_distances  # Broadcasting
            sum_squared_diff = np.sum(diff_matrix ** 2, axis=1)  # Shape: (n_subjects,)
            
            # Compute log probabilities
            log_probs = -sum_squared_diff / (2 * sigma ** 2)
            
            # Convert to probabilities (numerically stable)
            max_log_prob = np.max(log_probs)
            probs = np.exp(log_probs - max_log_prob)
            obs_probs[:, state_idx] = probs / np.sum(probs)
        
        return obs_probs
    
    def transition(self, known_state: KnownState, action: Tuple[int, ...]) -> KnownState:
        """
        Update known state after taking a quarter of courses.
        """
        return known_state + (action,)