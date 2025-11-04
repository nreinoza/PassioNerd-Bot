import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

UNIT_MIN = 12
UNIT_MAX = 20
PREREQ_MIN = 3
MAJOR_MIN = 40
GRAD_REWARD = 1000
ENJOYMENT_SCALE = 10
COSINE_ENJOYMENT = True

class Game():
    def __init__(self, csv: str, true_pref: np.array):
        # Main courses database
        courses = pd.read_csv(csv)
        known_cols = ['id', 'name', 'major', 'code', 'units']
        embedding_cols = [col for col in courses.columns if col not in known_cols]
        courses['embedding'] = courses[embedding_cols].apply(lambda row: row.to_numpy(dtype=np.float32), axis=1)
        courses = courses.drop(columns=embedding_cols)
        self.courses = courses.set_index('id')

        assert(true_pref.shape[0] == len(embedding_cols))
        self.true_pref = true_pref
        
        # Current state: units, quarter #, TODO belief distribution, classes taken
        self.active_units = 0
        self.total_units = 0
        self.active_courses = set()
        self.active_quarter = 1
        self.courses_taken = set()
        self.total_reward = 0.0

    def prereqs_covered(self, course_id) -> bool:
        """
        Returns a boolean saying whether or not the prereqs for this class are covered,
        using self.courses_taken (a list of course_ids).
        """
        if course_id not in self.courses.index:
            raise ValueError(f"course ID {course_id} not found in main course database.")
        
        # 1. Look up the target class's major and code
        target_class = self.courses.loc[course_id]
        required_major = target_class['major']
        target_code = target_class['code']
        
        # 2. Get the DataFrame of ALL classes taken by the student
        # This is efficient because we are only looking up the rows by their index (ID).
        classes_taken_df = self.courses.loc[self.courses_taken]
        
        # 3. Filter the taken classes by the required major
        major_classes_taken = classes_taken_df[classes_taken_df['major'] == required_major]
        
        # --- Rule 1: Class code > 110 and <= 200 ---
        if 110 < target_code <= 200:
            # Need at least 3 courses taken from that major
            required_count = 3
            return len(major_classes_taken) >= required_count
        
        # --- Rule 2: Class code > 200 ---
        elif target_code > 200:
            # Need at least 3 courses taken from that major with code > 110
            required_count = 3
            
            # Filter the major classes taken further by code > 110
            advanced_major_classes_taken = major_classes_taken[major_classes_taken['code'] > 110]
            
            return len(advanced_major_classes_taken) >= required_count
        
        # --- Default Case: Class code <= 110 ---
        # No complex prerequisites are required for lower-level courses
        return True
  
    def enjoyment(self, course_ids) -> dict[str, float]:
        enjoyments = {}
        if not COSINE_ENJOYMENT:
            for course_id in course_ids:
                res = input(f"Please give the class {self.courses.loc[course_id, 'name']} an enjoyment score between 0.0 and 10.0")
                res = int(res)
                if (res < 0 or res > 10):
                    ValueError("Hey why did you do that? :(")
                enjoyments[course_id] = res
            return enjoyments
                

        std_dev = 0.1
        # for each course_id
        norm_true_pref = np.linalg.norm(self.true_pref)

        # for each course_id
        for course_id in course_ids:
            # Get the course embedding
            course_embedding = self.courses.loc[course_id, 'embedding']
            norm_course_emb = np.linalg.norm(course_embedding)

            # 1. compute cosine similarity between self.true_pref and this courses' embedding
            if norm_course_emb == 0:
                ValueError(f"Course id {course_id} has embedding norm 0 (?)")
            else:
                dot_product = np.dot(self.true_pref, course_embedding)
                cosine_similarity = dot_product / (norm_true_pref * norm_course_emb)

            # 2. create gaussian with this mean and the std_dev above
            # 3. Take 1 sample from this gaussian. This is the enjoyment.
            enjoyment_sample = np.random.normal(loc=cosine_similarity, scale=std_dev)
            
            # 4. Add course_id : enjoyment pair to the enjoyments dict
            enjoyments[course_id] = (enjoyment_sample + 1) * (ENJOYMENT_SCALE / 2)
            
        return enjoyments

    def course_rewards(self, enjoyment_dict) -> float:

        total = 0.0
        total_units = 0
        for course_id, enjoyment in enjoyment_dict.items():
            total_units += self.courses.loc[course_id, 'units']
            total += self.courses.loc[course_id, 'units'] * enjoyment
        total /= total_units
        total *= (1.6 - total_units / 20)   # POSSIBLY MODIFY HEAVY QUARTER PENALTY
        
        return total

    def closest_takeable_class(self, target_embedding: np.array) -> str | None:
        """
        Finds the course ID of the closest available and takeable class based on
        the target embedding and game state constraints.

        Constraints:
        1. Units: self.active_units + course_units <= UNIT_MAX
        2. Taken: Class not already in self.courses_taken
        3. Prereqs: Prereqs are covered for the class
        """
        # Start with all available courses (not currently active)
        # Note: We include self.active_courses just in case this is called mid-selection,
        # but typically this should be an empty list when selecting the first class.
        available_courses = self.courses[~self.courses.index.isin(self.courses_taken + self.active_courses)].copy()

        # 1. Filter by Unit Constraint
        # Units for the course being considered (current row) + active_units <= UNIT_MAX
        unit_mask = available_courses['units'].apply(
            lambda units: self.active_units + units <= UNIT_MAX
        )
        filtered_courses = available_courses[unit_mask]

        if filtered_courses.empty:
            return None # No class satisfies the unit constraint

        # 2. Filter by Prerequisite Constraint
        # Use a list comprehension or apply to check prereqs_covered for each course ID
        prereq_mask = filtered_courses.index.map(self.prereqs_covered)
        
        takeable_courses = filtered_courses[prereq_mask]

        if takeable_courses.empty:
            return None # No class satisfies all constraints

        # --- Distance Calculation (Efficient Vectorized Operation) ---
        
        # 3. Create a 2D matrix of all embeddings for the takeable courses
        # np.stack efficiently converts the Series of NumPy arrays into a matrix
        embedding_matrix = np.stack(takeable_courses['embedding'].values)
        
        # 4. Calculate the Euclidean distance (L2 norm) between the target embedding
        # and every embedding in the matrix. This is highly optimized by NumPy.
        # axis=1 ensures the distance is calculated row-wise (i.e., for each course)
        distances = np.linalg.norm(embedding_matrix - target_embedding, axis=1)
        
        # 5. Find the index of the minimum distance
        closest_index_in_mask = np.argmin(distances)
        
        # 6. Get the corresponding course ID from the takeable_courses DataFrame index
        closest_course_id = takeable_courses.iloc[closest_index_in_mask].name # .name returns the index value (the course_id)
        
        return closest_course_id
    
    class Action:
        def __init__(self, embedding=None):
            if embedding:
                self.is_commit = False
            else:
                self.is_commit = True
            self.embedding = embedding
            assert((self.is_commit and self.embedding) or (not self.is_commit and not self.embedding))
    
    class StepResult:
        def __init__(self, enjoyments=None, graduation=None, nothing_new=None):
            assert not (enjoyments == None and not graduation and not nothing_new)
            self.enjoyments = enjoyments
            self.graduation = graduation
            self.nothing_new = nothing_new

    def step(self, action: Action) -> StepResult:
        if action.is_commit:
            assert(self.active_units >= UNIT_MIN)

            enj = self.enjoyment(self.active_courses)
            # Rewards
            self.total_reward += self.course_rewards(enj)
            if self.active_quarter == 12 and self.courses.loc[self.classes_taken, 'major'].value_counts().max() > MAJOR_MIN:
                total += GRAD_REWARD

            # Update state
            self.courses_taken += self.active_courses
            self.active_courses = set()
            self.total_units += self.active_units
            self.active_units = 0
            self.active_quarter += 1

            if self.active_quarter > 12:
                return self.StepResult(graduation=True)
            
            return self.StepResult(enjoyments=enj)
        
        else:
            id = self.closest_takeable_class(action.embedding)
            
            if id == None:
                # No available classes (either bc units or prereqs or all were taken)
                if self.active_units < UNIT_MIN:
                    # WEIRD SHIT GOING ON
                    ValueError(f"Somehow no classes can be taken but we have less than 12 units... about to crash")
                return self.step(self.Action()) # commit

            # Update state
            self.active_courses.add(id)
            self.active_units += self.courses.loc[id, 'units']
            return self.StepResult(nothing_new=True)

    def take_action(self, observations: dict[str: float]):
        return self.Action() # WILL ALWAYS COMMIT FOR NOW

    def run(self):
        observations = {}
        while True:
            # TAKE ACTION
            action = self.take_action(observations)
            res = self.step(action)
            
            if res.graduation:
                break
            elif res.nothing_new:
                continue
            else:
                observations = res.enjoyments
                # Update the belief system using these observations

