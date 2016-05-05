'use strict';

angular.module('myApp.home', ['ngRoute', 'angularFileUpload'])

.config(['$routeProvider', function($routeProvider) {
  $routeProvider.when('/home', {
    templateUrl: 'assets/home/home.html',
    controller: 'homeCtrl'
  });
}])

.controller('homeCtrl', ['$scope', '$http', 'FileUploader', function($scope, $http, FileUploader) {
    $scope.uploadError = "";
    $scope.status = 'IDLE';
    $scope.trainerType = "ridge";

    $scope.logs = [];
    //update predictor status
    var getConsoleLog = function() {
        $http.get('/get_tasks')
            .success(function(data) {
                $scope.logs = data;
            })
    }

    getConsoleLog();


    $scope.uploader = new FileUploader({
        url: '/upload_training_data',
        onWhenAddingFileFailed: function(item, filter, options) {
            $scope.uploadError = "An error occurred while uploading file"
        },
        onSuccessItem: function(item, response, status, headers) {
//            $scope.uploader.clearQueue();
            $scope.uploadError = ""
            alert("Successfully uploaded");
            getConsoleLog();
        },
        onErrorItem: function(item, response, status, headers) {
            $scope.uploadError = "An error occurred while uploading file"
        }
    });

    jQuery("input#fileId").change(function () {
        $scope.uploader.clearQueue();
    });

    $scope.triggerTraining = function() {
        $scope.status = 'Training in Progress';

        $http.post('/train_and_generate', { learner: $scope.trainerType })
            .success(function(res) {
                $scope.status = 'IDLE';
                getConsoleLog();
            })
            .error(function(err) {
                $scope.status = 'IDLE';
                alert('An error occurred');
            })
    };

    $scope.updatePricelistDB = function() {
        $scope.status = 'Updating DB';

        $http.post('/save_predicted_prices', {})
            .success(function(res) {
                $scope.status = 'IDLE';
                getConsoleLog();
            })
            .error(function(err) {
                $scope.status = 'IDLE';
                alert('An error occurred');
            })
    };
}]);