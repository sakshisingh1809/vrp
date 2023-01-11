# VRP

Repository with functions to do VRP and VRP-GA-ML operations.

Developer: Sakshi Singh (sakshisingh1809@gmail.com)

---

---

This python project is a research oriented projected that tries to solve the capacitated vehicle routing problem instances using the genetic algorithm.

The Capacitated Vehicle Routing Problem (CVRP) is a combinatorial optimiza-
tion problem that aims to find the best route for a group of vehicles to take in
order to serve a group of customers while meeting their needs and abiding by spe-
cific restrictions like vehicle capacity and time frames. The objective is to determine an ideal path for each vehicle that minimizes the overall distance traveled and takes into account the capacity restrictions of the vehicles.

The process of evolution in nature served as the inspiration for the search-
based optimization method known as genetic algorithms (GAs). They are
among the most potent and well-known optimization methods utilized in a va-
riety of practical situations. To create a population of solutions, GAs combine
selection, crossover, and mutation. From this population, the top candidates
are chosen to create the next generation. Until a solution is discovered that
satisfies the required requirements, the process is repeated.

The following steps are involved in this project (in this correct order):

1. Generation of dataset (first crucial step): The two files instancegeneration.py and datasetgeneration.py are used for this purposes. The first file generates an instance with two csv files: data.csv (containing actual data- customer coordinates (x,y) and their demands) and info.csv (containing the basic information about the data, ie, number of customers, minimum number of vehicles used and capacity of the vehicle). Note here the capacity is kept homogenous, meaning all the vehicles have same capacity. Using the second file, different classes of data are generated (1000 instances in each class/folder). The first class contains customers ranging from 10 to 50, the second class from 50 to 100, the third from 100 to 150 and the last class ranging from 150 to 200 customers. 

2. Generation of dataset features: This is the final dataset that is loaded in training of the machine learning model that will be discussed in the following steps. The instance feature file is calculated once and is stored for further usage. The features evaluated for each instance are: Instance name, Number of customers, Vehicle capacity, Degree of capacity utilization/ minimum number of vehicles, Standard deviation of demand, Average number of customers per vehicle route (no. of customers/ no. of vehicles), Average distance between each pair of customers, Average distance from customer to depot (radius of vrp instance), Variance in distance from customer to depot and Variance in pairwise distance between customers.

3. Solving the CPLEX model: In this step, each data.csv and info.csv file for each instance in each class is loaded in the cplex model and the vehicle routing problem is solved, generating a solution.sol and solutiongraph.png file in each of the respective folders of the instances. Along with this, the solution features are calculated and stored in a single file called SolutionsCPLEX.csv under the featuresdata/solutionfeatures folder. This step uses two files cplexmodel.py and visualize.py.

4. Generating the solution features for CPLEX solutions: The SolutionsCLPEX.csv contains 10 features related to the solutions solved using the Exact solver. These features are described: Instance name, Solver (Exact/VRP_GA/VRP_GA_ML, here Exact solver), Average distance between depot to directly-connected customers, Average distance between routes, Variance in number of customers per route, Longest distance between two connected customers, per route, Average depth per route, Number of routes, Cost and Optimality (0/1).

5. Loading data from VRP-GA program solved in C++ (in a seperate repository on my git). Different solution files from the VRP-GA program are loaded, the two most important are SolutionsGAgap2.csv and SolutionsGAgap4.csv which contains all the solutions features for all the 4000 instances in our original dataset (but the size of these csv files are 8000 rows each, as each instance has two solutions: one optimal solution row and another non-optimal solution row).

6. Converting all csv files to excel: convert all the files to excel to be loaded into the machine learning model using the csvtoexcel.py file.

7. Machine Learning training: The solution files for both the algorithms (CPLEX as well as GA) is merged with instance files. This final file is passed to ML models: Random forests and Support vector machines and accuracy for each model for each class is calculated seperately. The training on this 8000 solutions is performed once.

8. VRP-GA-ML model: The final step of the project is to integrate this trained model of machine learning into the genetic algorithm. In GA, in each iteration, the fitness of the population to make it to the next generation is calculated using a mathematical formula (highest fitness). But we are trying to replace this fitness value by the machine learning model with prior knowledge of the instances and the solutions. This knoweledge will help make a better decision for a chromosome in a GA whether it should be passed to the next generation or not. The input to this model is a set of solutions(child chromosomes) generated using the VRP-GA algorithm (in C++ in another repository) and result (0/1) indicate whether the solutions(child chromosomes) will pass to the next generation or not. This result file is passed back to the C++ program which solves VRP-GA problem.