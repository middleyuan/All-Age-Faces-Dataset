import pandas as pd
import random
import math

class Person(object):

    gender_list = ['F','M']
    age_list = ['[0, 3]','[4, 7]','[8, 14]','[15, 24]','[25, 37]','[38, 47]','[48, 59]','[60, 100]']

    def __init__(self, gender=0, age=0, second_best_age=0, ratio=0):
        self.gender_index = gender
        self.age_index = age
        self.gender = self.gender_list[self.gender_index]
        self.age = self.age_list[self.age_index]
        self.second_best_age_index = second_best_age
        self.second_best_age = self.age_list[self.second_best_age_index]
        self.ratio = ratio

class People(object):

    def __init__(self, age_predictions, age_second_predictions, gender_predictions, ratios):
        self.age_predictions = age_predictions
        self.age_second_predicitons = age_second_predictions
        self.gender_predictions = gender_predictions
        self.ratios = ratios
        self.person_list = []

        for (age_prediction, age_second_prediction, gender_prediction, ratio) in zip(age_predictions, age_second_predictions, gender_predictions, ratios):
            person = Person(gender_prediction, age_prediction, age_second_prediction, ratio)
            self.person_list.append(person)


class QuestionPosing(object):

    question_type = ['General', 'Couple', 'Family', 'Solo', 'Brothers-and-Sisters', 'Colleagues', 'Classmates']

    def __init__(self, path, people):
        self.template = pd.read_csv(path)
        self.people = people
        self.type = 0

    def ask(self):

        if len(self.people.person_list) == 1:
            self.type = 3 # Solo
            tmp = self.template[self.question_type[self.type]]
        elif len(self.people.person_list) == 2:
            if self.people.person_list[0].gender_index != self.people.person_list[1].gender_index and abs(self.people.person_list[0].age_index-self.people.person_list[1].age_index) <= 1:
                self.type = 1 # Couple
                tmp = self.template[self.question_type[self.type]]
            elif self.people.person_list[0].gender_index == self.people.person_list[1].gender_index and abs(self.people.person_list[0].age_index-self.people.person_list[1].age_index) <= 1:
                user_input = input('你們是什麽關係呢： 1. 夫妻 2. 兄弟姊妹 3. 同事 4. 同學 5. 家人 6. 都不是? ')
                if user_input == '1':
                    self.type = 1 # couple
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '2':
                    self.type = 4 # Brothers-and-Sisters
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '3':
                    self.type = 5 # Colleagues
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '4':
                    self.type = 6 # Classmates
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '5':
                    self.type = 2 # Family
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '6':
                    self.type = 0 # General
                    tmp = self.template[self.question_type[self.type]]
                else:
                    self.type = 0 # General
                    tmp = self.template[self.question_type[self.type]]
            else:
                self.type = 0 # General
                tmp = self.template[self.question_type[self.type]]

        elif len(self.people.person_list) >= 3:
            age = sorted(range(len(self.people.age_predictions)), key=lambda k: self.people.age_predictions[k])

            if self.people.age_predictions[age[len(age) - 1]] - self.people.age_predictions[age[0]] >= 2:
                self.type = 2 # Family
                tmp = self.template[self.question_type[self.type]]
            else: 
                user_input = input('你們是什麽關係呢： 1. 兄弟姊妹 2. 同事 3. 同學 4. 家人 5. 都不是? ')
                if user_input == '5':
                    self.type = 0 # General
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '1':
                    self.type = 4 # Brothers-and-Sisters
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '2':
                    self.type = 5 # Colleagues
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '3':
                    self.type = 6 # Classmates
                    tmp = self.template[self.question_type[self.type]]
                elif user_input == '4':
                    self.type = 2 # Family
                    tmp = self.template[self.question_type[self.type]]
                else:
                    self.type = 0 # General
                    tmp = self.template[self.question_type[self.type]]               

        self.questions = []
        for i in tmp:
            if not pd.isna(i):
                self.questions.append(i)

        shuffled_index = list(range(len(self.questions)))
        random.seed(random.randint)
        random.shuffle(shuffled_index)

        return self.questions[shuffled_index[0]]

        

