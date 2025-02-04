# AI Course Designer

## Overview

This project is a **Course Recommendation System** designed to help users identify the most relevant courses based on their learning goals or career aspirations. The system leverages **GPT-4** to extract and refine skills required for a user's input, and then uses **TF-IDF** and **BERT-based cosine similarity** to recommend courses from a dataset. The system also optimizes the course list by removing redundancies and sorting courses in a logical learning path.

The project is divided into several key components, each of which is explained in detail below. The code is modular, making it easy to extend or modify for different use cases.

---

## Table of Contents

1. [Project Flow](#project-flow)
2. [Code Explanation](#code-explanation)
   - [Step 1: Generate Skills List](#step-1-generate-skills-list)
   - [Step 2: Extract Relevant Skills from User Input](#step-2-extract-relevant-skills-from-user-input)
   - [Step 3: Extract Python List from GPT-4 Response](#step-3-extract-python-list-from-gpt-4-response)
   - [Step 4: Refine Skills List Based on User Feedback](#step-4-refine-skills-list-based-on-user-feedback)
   - [Step 5: Create TF-IDF Matrix](#step-5-create-tf-idf-matrix)
   - [Step 6: Convert Skills List to String](#step-6-convert-skills-list-to-string)
   - [Step 7: Find Similar Courses Using Cosine Similarity](#step-7-find-similar-courses-using-cosine-similarity)
   - [Step 8: Remove Overlapping Courses](#step-8-remove-overlapping-courses)
   - [Step 9: Sort Courses in Logical Order](#step-9-sort-courses-in-logical-order)
   - [Step 10: Generate Course Details](#step-10-generate-course-details)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

---

## Project Flow

1. **User Input**: The user provides a learning goal or career aspiration.
2. **Skill Extraction**: GPT-4 extracts relevant skills from the user's input.
3. **Skill Refinement**: The user can refine the list of skills by removing or modifying entries.
4. **Course Matching**: The system matches the refined skills to courses using **TF-IDF** and **BERT-based cosine similarity**.
5. **Course Optimization**: Overlapping courses are removed, and the remaining courses are sorted in a logical learning path.
6. **Course Details**: The system generates detailed information about each recommended course, including its relevance to the user's goal.

---

## Code Explanation

### Step 1: Generate Skills List

This function sends a course title to GPT-4 and retrieves a list of skills required to complete the course.

```python
def get_skills_from_gpt4(course_title):
    """
    Sends the course title to GPT-4 and asks for the skills required.
    Returns a comma-separated string of skills.
    """
    prompt = f"""
    You are an expert in analyzing course titles and identifying the skills required to complete the course.
    Based on the course title below, provide a list of skills that a learner must have or will acquire.
    Return only a comma-separated list of skills, nothing else.

    Course Title: {course_title}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Limit the response length
            temperature=0.5  # Control creativity (lower = more deterministic)
        )
        skills = response.choices[0].message.content.strip()
        return skills
    except Exception as e:
        print(f"Error fetching skills for '{course_title}': {e}")
        return None
```

---

### Step 2: Extract Relevant Skills from User Input

This function takes user input, sends it to GPT-4, and extracts a list of relevant skills needed to achieve the user's goal.

```python
def extract_relevant_skills(user_input):
    prompt = f"""
    You are a career and learning advisor. A user has shared the following issue or goal:

{user_input}

Your task is to:
1) Understand the user's issue or goal.
2) Identify and extract a list of **technical and domain-specific skills** that the user needs to learn or improve to solve the issue or achieve the goal.
3) Exclude general or assumed skills (e.g., critical thinking, communication, problem-solving) that are not specific to the domain.
4) Ensure each skill is listed as a **single, concise term or phrase** (e.g., "Hadoop" instead of "Knowledge of Hadoop & Spark").
5) Return the list of skills in a clear and concise format.

Provide the output as a Python list of skills, like this:
["Skill 1", "Skill 2", "Skill 3"]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  
            temperature=0.5  
        )

        response_content = response.choices[0].message.content.strip()

        # Extracting the list of skills from the response
        list_match = re.search(r'\[.*\]', response_content)
        if list_match:
            list_string = list_match.group(0)
            try:
                skills_list = eval(list_string)
                if isinstance(skills_list, list):  # Ensuring it's a valid list
                    return skills_list
                else:
                    print("Error: GPT-4 did not return a valid list of skills.")
                    return None
            except Exception as e:
                print(f"Error parsing GPT-4 response into a list: {e}")
                return None
        else:
            print("Error: GPT-4 response does not contain a valid list.")
            return None

    except Exception as e:
        print(f"Error fetching skills from GPT-4: {e}")
        return None
```

---

### Step 3: Extract Python List from GPT-4 Response

This function extracts a Python list from GPT-4's response, even if it includes extra text.

```python
def extract_list_from_response(response):
    """
    Extracts a Python list from GPT-4's response, even if it includes extra text.
    """
    try:
        # Finding the start and end of the Python list
        start_index = response.find('[')
        end_index = response.find(']') + 1

        list_str = response[start_index:end_index]

        #Evaluating the string into a Python list
        return eval(list_str)
    except Exception as e:
        print(f"Error extracting list from response: {e}")
        return None
```

---

### Step 4: Refine Skills List Based on User Feedback

This function allows the user to refine the list of skills by removing or modifying entries.

```python
def final_relevant_skills(user_input, skills):
    prompt = f"""
You are assisting a user who has provided input: "{user_input}" and a list of relevant skills: {skills}. Based on their input, you need to refine the list of relevant skills. Decide which skills to keep, remove, or modify in the list.

Your task is to carefully analyze the user's response and determine whether the list of relevant skills should remain unchanged or if certain skills should be removed or modified. Even if the user does not explicitly state their preferences, infer their intent based on the context, tone, and content of their response.

Ensure that your final output reflects the user's preferences accurately and provides a clear, updated list of relevant skills.

Provide the output as a Python list of skills, like this:
["Skill 1", "Skill 2", "Skill 3"]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500, 
            temperature=0.5  
        )
        response_content = response.choices[0].message.content.strip()
        updated_skills = extract_list_from_response(response_content)
        if isinstance(updated_skills, list):  
            return updated_skills
        else:
            print("Error: GPT-4 did not return a valid list of skills.")
            return None

    except Exception as e:
        print(f"Error fetching skills from GPT-4: {e}")
        return None
```

---

### Step 5: Create TF-IDF Matrix

This function creates a TF-IDF matrix from the course titles in the dataset.

```python
ds = df.copy()

columns_to_remove = [
    "Unnamed: 0", "course_organization", "course_Certificate_type",
    "course_rating", "course_difficulty", "course_students_enrolled", "course_skills"
]

ds.drop(columns=columns_to_remove, inplace=True, errors="ignore")

from sklearn.feature_extraction.text import TfidfVectorizer

course_titles = ds["course_title"].astype(str).str.lower()  

# Initializing the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Computing the matrix
tfidf_matrix = vectorizer.fit_transform(course_titles)

course_title_list = ds["course_title"].tolist()

print("TF-IDF Matrix shape:", tfidf_matrix.shape)
```

---

### Step 6: Convert Skills List to String

This function converts the list of skills into a single string with comma separation.

```python
skills_string = ', '.join(updated_skills)

final_updated_skills = f'"{skills_string}"'

print(final_updated_skills)
```

---

### Step 7: Find Similar Courses Using Cosine Similarity

This function finds similar courses using **TF-IDF** and **BERT-based cosine similarity**.

```python
def find_similar_course(input_text, tfidf_matrix, vectorizer, course_title_list, similarity_threshold=0.40):
    input_text = input_text.lower() 
    input_vector = vectorizer.transform([input_text]) 
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten() 

    top_indices = similarity_scores.argsort()[::-1]  
    course_list = [course_title_list[i] for i in top_indices if similarity_scores[i] > similarity_threshold]

    return course_list

final_course_list = find_similar_course(final_updated_skills, tfidf_matrix, vectorizer, course_title_list)

print("Courses found:", final_course_list)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight BERT model

def find_similar_course_bert(input_text, course_title_list, similarity_threshold=0.40):
    input_text = input_text.lower()
    input_embedding = model.encode([input_text])
    course_embeddings = model.encode(course_title_list)

    similarity_scores = cosine_similarity(input_embedding, course_embeddings).flatten()
    course_list = [course_title_list[i] for i, score in enumerate(similarity_scores) if score > similarity_threshold]

    return course_list

final_course_list = find_similar_course_bert(final_updated_skills, course_title_list)
print("Courses found (BERT):", final_course_list)
```

---

### Step 8: Remove Overlapping Courses

This function removes overlapping courses from the list.

```python
def extract_relevant_skills2(final_course_list):
    prompt = f"""
    You are an expert in analyzing and optimizing course lists. I have a list of courses {final_course_list}, which contains multiple courses. Each course has a title. Your task is to:

    1) Analyze the courses in final_course_list and identify any overlapping content between them.

    2) For courses with significant overlap, determine which course covers the majority of the content and retain that one. Remove the others to avoid redundancy.

    3) Ensure that any unique courses (those without significant overlap) are kept in the list.

    4) Return a refined and optimized version of final_course_list that minimizes redundancy while preserving all unique and essential content.

    5) Provide the final list of courses in a clear and organized format.

    **Important:** Your output must be a Python list of course titles, like this, always in this format:
    ["Course Title 1", "Course Title 2", "Course Title 3"]

    **Do not include any additional text or explanations in your response. Only return the Python list.**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  
            temperature=0.5  
        )

        response_content = response.choices[0].message.content.strip()

        try:
            response_content = response_content.strip()
            if response_content.startswith("```python"):
                response_content = response_content[len("```python"):].strip()
            if response_content.endswith("```"):
                response_content = response_content[:-len("```")].strip()

            skills_list = ast.literal_eval(response_content)
            if isinstance(skills_list, list):  
                return skills_list
            else:
                print("Error: GPT-4 did not return a valid list of skills.")
                return None
        except Exception as e:
            print(f"Error parsing GPT-4 response into a list: {e}")
            print("Raw Response Content:", response_content) 
            return None

    except Exception as e:
        print(f"Error fetching skills from GPT-4: {e}")
        return None

final_response_list = extract_relevant_skills2(final_course_list)

print("Optimized Course List:", final_response_list)
```

---

### Step 9: Sort Courses in Logical Order

This function sorts the courses in a logical learning path.

```python
def sort_courses(final_response_list):
    prompt = f"""
    You are an expert career and learning advisor. A user has provided a list of courses they need to complete, and your task is to analyze and sort these courses in the order they should be completed. The goal is to create a logical learning path that considers the following factors:

1. **Difficulty Level:** Easier courses should generally come before more advanced ones.
2. **Prerequisites:** Courses that build on foundational knowledge should come after those that teach the basics.
3. **Logical Progression:** Courses should flow naturally from one to the next, ensuring the user gains skills in a structured way.

Here is the list of courses:
{final_course_list}

Your task is:
1) Analyze the courses and determine the optimal order in which the user should complete them.
2) Return the updated `final_response_list` with the courses sorted in the recommended order.

**Important:**
- Your output must be **only a Python list of course titles**, sorted in the recommended order, like this:
["Course Title 1", "Course Title 2", "Course Title 3"]

- **Do not include any additional text, explanations, or numbering in your response. Only return the Python list.**
    """

    try:
        # Send the prompt to GPT-4
        response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Adjust as needed
            temperature=0.5  # Control creativity (lower = more deterministic)
        )

        response_content = response.choices[0].message.content.strip()

        optimized_course_list = extract_list_from_response(response_content)
        if isinstance(optimized_course_list, list):  
            return optimized_course_list
        else:
            print("Error: GPT-4 did not return a valid list of skills.")
            return None

    except Exception as e:
        print(f"Error fetching skills from GPT-4: {e}")
        return None

final_response_list = sort_courses(final_response_list)

print("Optimized Course List:", final_response_list)
```

---

### Step 10: Generate Course Details

This function generates detailed information about each recommended course.

```python
def generate_course_details(final_response_list, df):
    # Initialize an empty list to store the results
    result_list = []

    # Iterate through each course_title in final_response_list
    for course_title in final_response_list:
        # Find the row in df where course_title matches
        matched_row = df[df['course_title'] == course_title]

        # Check if a match was found
        if not matched_row.empty:
            # Extract the relevant columns as a list
            course_details = [
                matched_row['course_title'].values[0],
                matched_row['course_organization'].values[0],
                matched_row['course_rating'].values[0],
                matched_row['course_difficulty'].values[0],
            ]

            # Append the course_details list to result_list
            result_list.append(course_details)
        else:
            # Handle the case where the course_title is not found in df
            print(f"Warning: Course title '{course_title}' not found in the DataFrame.")

    # Return the result_list
    return result_list

# Call the function
course_details_list = generate_course_details(final_response_list, df)

# Print the result
print("Generated Course Details:")
for course in course_details_list:
    print(course)
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/course-recommendation-system.git
   cd course-recommendation-system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

---

## Usage

1. Run the script:
   ```bash
   python main.py
   ```

2. Follow the prompts to input your learning goal or career aspiration.

3. The system will output a list of recommended courses, sorted in a logical learning path.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
