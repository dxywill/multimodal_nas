
from os import path

import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm


def load_data_with_features_continious(file_name):

    pickle_data = {}
    students = {}
    students_list = []
    skills = []
    users_id = []
    max_steps = -1
    min_steps = float('inf')
    features = ['assignment_id', 'assistment_id', 'problem_id', 'user_id', 'original', 'correct', 'attempt_count',
                'ms_first_response',
                'skill_id', 'hint_count', 'hint_total', 'first_action', 'bottom_hint']

    #selected_features = ['assignment_id', 'assistment_id', 'problem_id', 'original', 'correct', 'attempt_count',
    #                     'ms_first_response',
    #                     'skill_id', 'hint_count', 'hint_total', 'first_action', 'bottom_hint']

    selected_features = ['skill_id', 'correct', 'user_id', 'ms_first_response', 'attempt_count', 'hint_total']
    #selected_features_d = ['skill_id', 'correct', 'user_id', 'ms_first_response_d', 'attempt_count_d']
    if not path.exists('students.pickle'):
        print('Pickle file not found, creating one...')
        all_data = pd.read_csv(file_name, encoding='ISO-8859-1')
        filtered_data = all_data[features]
        filtered_data = filtered_data[filtered_data['ms_first_response'] > 0]
        filtered_data['ms_first_response'] = (filtered_data['ms_first_response'] - filtered_data['ms_first_response'].min()) \
            / (filtered_data['ms_first_response'].max() - filtered_data['ms_first_response'].min())

        filtered_data['attempt_count'] = (filtered_data['attempt_count'] - filtered_data[
            'attempt_count'].min()) / (filtered_data['attempt_count'].max() - filtered_data['attempt_count'].min())

        filtered_data['hint_total'] = (filtered_data['hint_total'] - filtered_data[
            'hint_total'].min()) / (filtered_data['hint_total'].max() - filtered_data['hint_total'].min())
        filtered_data = filtered_data.fillna(-1)

        for index, row in tqdm(filtered_data.iterrows()):
            if row['skill_id'] == -1:
                continue
            if row['skill_id'] not in skills:
                skills.append(int(row['skill_id']))
            if row['user_id'] not in users_id:
                users_id.append(int(row['user_id']))
            if row['user_id'] in students:
                students[row['user_id']].append(row[selected_features].values.tolist())
            else:
                students[row['user_id']] = [row[selected_features].values.tolist()]

        # for user_id in users_id:
        #     students[user_id] = sorted(students[user_id], key=cmp_to_key(custom_sort), reverse=True)
        pickle_data['students'] = students
        pickle_data['skills'] = skills
        pickle_data['users_id'] = users_id
        pickling_on = open("students.pickle", "wb")
        pickle.dump(pickle_data, pickling_on)
        pickling_on.close()
    else:
        print('Load data from pickle file...')
        pickle_off = open("students.pickle", "rb")
        pickle_data = pickle.load(pickle_off)
        students = pickle_data['students']
        skills = pickle_data['skills']
        users_id = pickle_data['users_id']

    for user_id in users_id:
        if len(students[user_id]) > max_steps:
            max_steps = len(students[user_id])
        if len(students[user_id]) < min_steps:
            min_steps = len(students[user_id])
        if len(students[user_id]) <= 2:
            del students[user_id]
        else:
            students_list.append(students[user_id])

    print('Max step:', max_steps)
    print('Num of skills:', len(skills))
    print('Num of students:', len(users_id))

    return  np.array(students_list), skills, max_steps
