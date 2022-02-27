import re
from enum import Enum
import random

POPULATION_SIZE = 9
NUM_OF_ELITE_SCHEDULES = 1
SIZE_OF_RANDOM_POP_PART = 3
MUTATION_RATE = 0.1
MAX_GENERATION = 100


class Day(Enum):
    po = 1
    ut = 2
    st = 3
    ct = 4
    pa = 5


class Instructor:
    """
    attributes: name        - instructors name
                preference  - number in interval [0,1]
                              0   - I do not prefer this instructor
                              0.5 - I don't care about this instructor
                              1   - I really want this instructor
    """

    def __init__(self, preference, name):
        self._preference = preference
        self._name = name

    def get_name(self):
        return self._name

    def get_preference(self):
        return self._preference

    def __str__(self):
        return self._name


class Term:
    """
    attributes: day         - day in week
                start       - lesson start
                end         - lesson end
                instructor  - lesson instructor
    """

    def __init__(self, day: Day, start: float, end: float, instructor: Instructor):
        self._day = day
        self._start = start
        self._end = end
        self._instructor = instructor

    def get_day(self):
        return self._day

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

    def get_instructor(self):
        return self._instructor

    def __str__(self):
        return self._day + " " + str(self._start) + "-" + str(self._end) + " " + self._instructor.__str__()


class Terms:
    """
    attributes: term list - list of terms
    """

    def __init__(self):
        self._term_list = []

    def get_term_list(self):
        return self._term_list

    def add_term(self, term: Term):
        self._term_list.append(term)

    def __str__(self):
        return "\n".join(str(term) for term in self._term_list)


class Subject:
    """
    attributes: name    - subject name
                terms   - possible subject terms
    """

    def __init__(self, name, terms: Terms):
        self._name = name
        self._terms = terms

    def get_name(self):
        return self._name

    def get_terms(self):
        return self._terms

    def __str__(self):
        return self._name + "\n" + self._terms.__str__() + "\n"


class Data:
    """
    attributes: subject list - list of subjects
    """

    def __init__(self):
        self._subject_list = []

    def get_subjects(self):
        return self._subject_list

    def add_subject(self, subject):
        self._subject_list.append(subject)

    def __str__(self):
        return "\n".join(str(subject) for subject in self._subject_list)


class Schedule:
    """
    attributes: data
                fitness                 - -1 before counting
                                          number in interval [0,1]
                                          0 - worst solution
                                          1 - (one of the) best solution
                fitness is changed
                conflicts               - number of conflicts
                subject with one term   - subject containing just one term
    """

    def __init__(self):
        self._data = data
        self._fitness = -1
        self._fitness_is_changed = True
        self._conflicts = 0
        self._subject_with_one_term = []

    def get_data(self):
        return self._data

    def get_fitness(self):
        if self._fitness_is_changed:
            self._fitness = self.calculate_fitness()
            self._fitness_is_changed = False
        return self._fitness

    def get_number_of_conflicts(self):
        return self._conflicts

    def get_subjects(self, index=None):
        if index is None:
            return self._subject_with_one_term
        return self._subject_with_one_term[index]

    def set_subject(self, index, new_subject):
        self._fitness_is_changed = True
        self._subject_with_one_term[index] = new_subject

    def initialize(self):
        """ fill schedule with subjects

        :return: self
        """
        for subject in self._data.get_subjects():
            self._subject_with_one_term.append(Subject(subject.get_name(),
                                                       random.choice(subject.get_terms().get_term_list())))
        return self

    @staticmethod
    def _same_days(subject_a, subject_b):
        """ compares two days of subject term

        :param subject_a: subject to compare
        :param subject_b: subject to compare
        :return: bool
        """
        return subject_a.get_terms().get_day() == subject_b.get_terms().get_day()

    @staticmethod
    def _time_intersect(subject_a, subject_b):
        """ compares two lesson times of subject term

        :param subject_a: subject to compare
        :param subject_b: subject to compare
        :return: bool
        """
        return (subject_a.get_terms().get_start() <= subject_b.get_terms().get_end() and
                subject_b.get_terms().get_start() <= subject_a.get_terms().get_end())

    def calculate_fitness(self):
        """ calculates schedule fitness

        :return: number in interval [0,1]
        """
        preference_mean = 0
        self._conflicts = 0
        num_of_subjects = len(self._subject_with_one_term)
        for i in range(num_of_subjects - 1):
            preference_mean += self._subject_with_one_term[i].get_terms().get_instructor().get_preference()
            for j in range(i + 1, num_of_subjects):
                if self._same_days(self._subject_with_one_term[i], self._subject_with_one_term[j]):
                    if self._time_intersect(self._subject_with_one_term[i], self._subject_with_one_term[j]):
                        self._conflicts += 1
        preference_mean += self._subject_with_one_term[
            num_of_subjects - 1].get_terms().get_instructor().get_preference()
        preference_mean /= num_of_subjects
        return preference_mean / (1 + 2 * self._conflicts)

    def __str__(self):
        return "\n".join(subject.get_name() + "\t" + str(subject.get_terms())
                         for subject in self._subject_with_one_term)


class SchedulePopulation:
    """
    attributes: size        - population size (length of schedule list)
                data
                schedules   - list of schedules
    """

    def __init__(self, size):
        self._size = size
        self._data = data
        self._schedules = []
        for i in range(0, size):
            self._schedules.append(Schedule().initialize())

    def get_schedules(self, index=None):
        if index is None:
            return self._schedules
        return self._schedules[index]

    def set_schedule(self, index, new_schedule):
        self._schedules[index] = new_schedule

    def append_schedule(self, schedule):
        self._schedules.append(schedule)

    def sort_schedules(self):
        """ sort schedules in population by its fitness

        :return: sorted schedules
        """
        self._schedules.sort(key=lambda x: x.get_fitness(), reverse=True)

    def __str__(self):
        return "\n\n".join(str(schedule) + "\nfitness " + str(round(schedule.get_fitness(), 3)) +
                           "\tconflicts: " + str(schedule.get_number_of_conflicts()) for schedule in self._schedules)


class GeneticAlgorithm:
    def _crossover_population(self, population: SchedulePopulation):
        """ crossover non-elite part of population

        :param population:
        :return: crossovered population
        """
        crossover_pop = SchedulePopulation(0)
        for i in range(NUM_OF_ELITE_SCHEDULES):
            crossover_pop.append_schedule(population.get_schedules(i))
        for i in range(POPULATION_SIZE - NUM_OF_ELITE_SCHEDULES):
            schedule_a = self._select_random_population_part(population).get_schedules(0)
            schedule_b = self._select_random_population_part(population).get_schedules(0)
            crossover_pop.append_schedule(self._crossover_schedule(schedule_a, schedule_b))
        return crossover_pop

    def _mutate_population(self, population):
        """ mutate non-elite part of population

        :param population:
        :return: mutated population
        """
        for i in range(NUM_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            self._mutate_schedule(population.get_schedules(i))
        return population

    @staticmethod
    def _crossover_schedule(schedule_a, schedule_b):
        """ crossover two schedules

        :param schedule_a: schedule to crossover
        :param schedule_b: schedule to crossover
        :return: crossovered schedule
        """
        crossover_schedule = Schedule().initialize()
        for i in range(len(crossover_schedule.get_subjects())):
            if random.choice([True, False]):
                crossover_schedule.set_subject(i, schedule_a.get_subjects(i))
            else:
                crossover_schedule.set_subject(i, schedule_b.get_subjects(i))
        return crossover_schedule

    @staticmethod
    def _mutate_schedule(mutate_schedule):
        """ mutate schedule

        :param mutate_schedule: schedule to mutate
        :return: mutated schedule
        """
        schedule = Schedule().initialize()
        for i in range(len(mutate_schedule.get_subjects())):
            if MUTATION_RATE > random.random():
                mutate_schedule.set_subject(i, schedule.get_subjects(i))
        return mutate_schedule

    @staticmethod
    def _select_random_population_part(population):
        """ randomly select part of population

        :param population:
        :return: population part
        """
        population_part = SchedulePopulation(0)
        for i in range(SIZE_OF_RANDOM_POP_PART):
            population_part.append_schedule(population.get_schedules(random.randrange(0, POPULATION_SIZE)))
            population_part.sort_schedules()
        return population_part

    def evolve(self, population):
        return self._mutate_population(self._crossover_population(population))


def time_to_float(time):
    """ convert time in HH:MM format to float

    :param time: HH:MM
    :return: float time
    """
    x = time.split(sep=":")
    return round(int(x[0]) + int(x[1]) / 60, 2)


def subject_from_line(line):
    """ convert line in format SUBJECT_NAME - DAY START_TIME-END_TIME INSTRUCTOR_NAME INSTRUCTOR_PREFERENCE, DAY ...
        to subject

    :param line: line from file
    :return: subject
    """
    split_line = line.split()
    terms = Terms()
    for i in range(2, len(split_line), 4):
        instructor = Instructor(float(re.sub(r",", '', split_line[i + 3])), split_line[i + 2])
        times = split_line[i + 1].split(sep="-")
        term = Term(day=split_line[i], start=time_to_float(times[0]), end=time_to_float(times[1]),
                    instructor=instructor)
        terms.add_term(term)
    subject = Subject(split_line[0], terms)
    return subject


def read_file(path):
    """ read file line by line

    :param path: file path
    :return: data
    """
    file1 = open(path, 'r')
    lines = file1.readlines()
    count = 0
    dataa = Data()
    for line in lines:
        subject = subject_from_line(line)
        dataa.add_subject(subject)
        count += 1
    return dataa


def run_and_print_populations():
    """ runs genetic algorithm to find best schedule and prints all populations """
    generation_num = 0
    pop = SchedulePopulation(POPULATION_SIZE)
    pop.sort_schedules()
    print("\n\tGeneration", generation_num)
    print(pop.__str__())
    genetic_alg = GeneticAlgorithm()
    while generation_num < MAX_GENERATION and pop.get_schedules(0).get_fitness() != 1:
        generation_num += 1
        print("\nGeneration", generation_num)
        pop = genetic_alg.evolve(pop)
        pop.sort_schedules()
        print(pop.__str__())


def run_and_print_best_schedule():
    """ runs genetic algorithm to find best schedule and prints best schedule """
    generation_num = 0
    pop = SchedulePopulation(POPULATION_SIZE)
    pop.sort_schedules()
    genetic_alg = GeneticAlgorithm()
    while generation_num < MAX_GENERATION and pop.get_schedules(0).get_fitness() != 1:
        generation_num += 1
        pop = genetic_alg.evolve(pop)
        pop.sort_schedules()
    print("\nGeneration", generation_num)
    print(pop.get_schedules(0).__str__() + "\nfitness " + str(round(pop.get_schedules(0).get_fitness(), 3)) +
          "\tconflicts: " + str((pop.get_schedules(0).get_number_of_conflicts())))


data = read_file("data.txt")  # sample data
run_and_print_best_schedule()
