'use strict';

angular.module('myApp.home', ['ngRoute', 'angularFileUpload'])

.config(['$routeProvider', function($routeProvider) {
  $routeProvider.when('/home', {
    templateUrl: 'assets/home/home.html',
    controller: 'homeCtrl'
  });
}])

.controller('homeCtrl', function($scope, $http, FileUploader, APIService) {
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

    var getFileList = function() {
        APIService.getFileList()
            .then(function(fileList) {
                $scope.files = fileList;
            });
    }

    $scope.deleteFile = function(file) {
        APIService.deleteFile(file)
            .then(function() {
                alert('Successfully deleted file');
            }, function(err) {
                alert('Failed to delete file');
            });
        getFileList();
    };

    getFileList();

    $scope.trainingFile = 'txn.csv';
    $scope.mappingFile = 'MappingPricer.csv';

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
            getFileList();
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
        APIService.triggerTraining($scope.trainerType, $scope.trainingFile, $scope.mappingFile)
            .then(function() {
                $scope.status = 'IDLE';
                getConsoleLog();
            },
            function(err) {
                $scope.status = 'IDLE';
                alert('An error occurred');
            });
    };

    $scope.vehicleType = 'car';
    $scope.updatePricelistDB = function() {
        $scope.status = 'Updating DB';

        $http.post('/save_predicted_prices', { vehicle_type: $scope.vehicleType })
            .success(function(res) {
                $scope.status = 'IDLE';
                getConsoleLog();
            })
            .error(function(err) {
                $scope.status = 'IDLE';
                alert('An error occurred');
            })
    };
});