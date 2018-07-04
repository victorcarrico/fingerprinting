import csv
import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from helpers import frange, distance
from PyRadioLoc.Pathloss.Models import Cost231Model

from constants import ERBS, EIRP

start_lat = -8.080
end_lat = -8.065
start_long = -34.91
end_long = -34.887


STEPS = {
    5: 0.000045,
    10: 0.00009,
    20: 0.00018
}

# 20 meters
step_long = 0.00018
step_lat = 0.00018


# MAP SIZE:
# largura: 1.667
# altura: 2.532


def calc_pathloss(d, model='svr'):
    if model == 'svr':
        svr = train_pathloss_simple_regression()
        return svr.predict(d)
    elif model == 'cost231':
        cost231_model = Cost231Model(1800)
        return cost231_model.pathloss(d)


class PathLoss:
    def __init__(self, model):
        self.model_name = model
        if model == 'svr':
            self.model = train_pathloss_simple_regression()
        elif model == 'cost231':
            self.model = Cost231Model(1800)

    def calc(self, d):
        if self.model_name == 'svr':
            return self.model.predict(d)
        elif self.model_name == 'cost231':
            return self.model.pathloss(d)


def construct_map():
    # cost231 = Cost231Model(1800)
    svr = train_pathloss_simple_regression()
    with open('map.csv', 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        for i, x in enumerate(frange(start_lat, end_lat, step_lat)):
            for y in frange(start_long, end_long, step_long):
                row = [x, y]
                for erb in ERBS:
                    d = distance((x, y), erb)
                    # pathloss_erb = cost231.pathloss(d)
                    pathloss_erb = svr.predict(d)
                    row.append(pathloss_erb[0])
                writer.writerow(row)
            print('col {}'.format(i))


def get_most_similar(grid_pathlosses, pathlosses_point):
    # return min(grid_pathlosses, key=lambda x: np.linalg.norm(x[2:] - pathlosses_point))
    return min(grid_pathlosses, key=lambda x: 1-(np.dot(x[2:], pathlosses_point)/(np.linalg.norm(x[2:])*np.linalg.norm(pathlosses_point))))
    # return min(grid_pathlosses, key=lambda x:sum(abs(a - b) for a,b in zip(x[2:],pathlosses_point)))


def bulk_pathloss():
    with open('data/testLoc.csv', 'r') as testloc:
        csvreader = csv.reader(testloc, delimiter=',')
        next(csvreader)
        errors = []
        for row in csvreader:
            point_target = (float(row[0]), float(row[1]))
            rssis = [float(rssi) for rssi in row[2:]]
            pathlosses = np.subtract(EIRP, rssis)
            with open('map.csv', 'r') as mapgrid_file:
                # mapreader = csv.reader(maploc, delimiter=',')
                # nearest_100 = get_n_nearest(mapreader, pathlosses[0], 100000)
                map_grid = np.loadtxt(mapgrid_file, delimiter=',')
                prediction_on_grid = get_most_similar(map_grid, pathlosses)
                point_prediction = prediction_on_grid[:2]
                error = distance(point_prediction, point_target)
                errors.append(error)
        return errors


def fingerprinting():
    with open('LocTest_v2.csv', 'r') as testloc:
        csvreader = csv.reader(testloc, delimiter=',')
        next(csvreader)
        # csvreader = [[-8.07752, -34.899162, -67.3, -24.7, -73.4, -67.2, -79.2666666666667, -80.6], ]
        errors = []
        prediction_locs = []
        point_targets = []
        for row in csvreader:
            point_target = (float(row[0]), float(row[1]))
            rssis = [float(rssi) for rssi in row[2:]]
            pathlosses = np.subtract(EIRP, rssis)
            # print('Pathloss Point: {}'.format(pathlosses))
            with open('map.csv', 'r') as mapgrid_file:
                # mapreader = csv.reader(maploc, delimiter=',')
                # nearest_100 = get_n_nearest(mapreader, pathlosses[0], 100000)
                map_grid = np.loadtxt(mapgrid_file, delimiter=',')
                prediction_on_grid = get_most_similar(map_grid, pathlosses)
                prediction_locs.append(prediction_on_grid[:2])
                point_targets.append(point_target)
                # print('Prediction on Grid: {}'.format(prediction_on_grid))
                point_prediction = prediction_on_grid[:2]
                error = distance(point_prediction, point_target)
                errors.append(error)
        print("Media: {}".format((sum(errors)/len(errors))))
        print('Melhor: {}'.format(min(errors)))
        return


def create_pathloss_csv():
    with open('data/medicoes.csv', 'r') as med_file:
        csvreader = csv.reader(med_file, delimiter=',')
        next(csvreader)
        with open('pathloss_data.csv', 'w') as path_file:
            writer = csv.writer(path_file, delimiter=',')
            for row in csvreader:
                point_loc = (float(row[0]), float(row[1]))
                line = []
                for i, erb_loc in enumerate(ERBS):
                    d_erb = distance(point_loc, erb_loc)
                    line.append(d_erb)
                    real_pathloss = 55.59 - float(row[i + 2])
                    line.append(real_pathloss)
                    writer.writerow(line)


def create_pathloss_csv_test():
    with open('data/testLoc.csv', 'r') as test_file:
        csvreader = csv.reader(test_file, delimiter=',')
        next(csvreader)
        with open('pathloss_data_test.csv', 'w') as path_file:
            writer = csv.writer(path_file, delimiter=',')
            for row in csvreader:
                point_loc = (float(row[0]), float(row[1]))
                line = []
                for i, erb_loc in enumerate(ERBS):
                    d_erb = distance(point_loc, erb_loc)
                    line.append(d_erb)
                    real_pathloss = 55.59 - float(row[i + 2])
                    line.append(real_pathloss)
                    writer.writerow(line)


def train_pathloss_simple_regression():
    with open('pathloss_data.csv', 'r') as path_file:
        X, y = np.loadtxt(path_file, delimiter=',', usecols=(0, 1), unpack=True)
        clf = SVR(C=1.0, epsilon=0.2)
        clf.fit(X.reshape(-1, 1), y)
    return clf


def test_pathloss_simple_regression():
    create_pathloss_csv_test()
    svr = train_pathloss_simple_regression()
    with open('pathloss_data_test.csv', 'r') as path_file:
        csvreader = csv.reader(path_file, delimiter=',')
        real_pathlosses = []
        prediction_pathlosses = []
        for row in csvreader:
            distance = row[0]
            pathloss_real = row[1]
            pathloss_prediction = svr.predict(distance)
            # print('Prediction: {}'.format(pathloss_prediction))
            # print('Real: {}\n'.format(pathloss_real))
            # print('----x----')
            prediction_pathlosses.append(pathloss_prediction)
            real_pathlosses.append(float(pathloss_real))
        mse = np.sqrt(
            mean_squared_error(real_pathlosses, prediction_pathlosses))
        return mse


def validate_pathloss_svr():
    """Validating the SVR model to predict the pathloss from all 6 ERBS
    using 10 Fold Cross Validation.
    """
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    linear_model = RandomForestRegressor()
    with open('pathloss_data.csv') as f:
        X, y = np.loadtxt(f, delimiter=',', usecols=(0, 1), unpack=True)
        kf = KFold(n_splits=10)
        mses = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # svr.fit(X_train.reshape(-1, 1), y_train)
            # y_predict = svr.predict(X_test.reshape(-1, 1))
            linear_model.fit(X_train.reshape(-1, 1), y_train)
            y_predict = linear_model.predict(X_test.reshape(-1, 1))
            mse = np.sqrt(mean_squared_error(y_test, y_predict))
            mses.append(mse)
            print('Train fold: {}'.format(train_index))
        return mses
