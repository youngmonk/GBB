'use strict';

angular.module('myApp.home', ['ngRoute', 'angularFileUpload'])

.config(['$routeProvider', function($routeProvider) {
  $routeProvider.when('/home', {
    templateUrl: 'assets/home/home.html',
    controller: 'homeCtrl'
  });
}])

.controller('homeCtrl', ['$scope', 'FileUploader', function($scope, FileUploader) {
    $scope.uploadError = "";

    $scope.uploader = new FileUploader({
        url: '/upload_training_data',
        onWhenAddingFileFailed: function(item, filter, options) {
            $scope.uploadError = "An error occurred while uploading file"
        },
        onSuccessItem: function(item, response, status, headers) {
//            $scope.uploader.clearQueue();
            $scope.uploadError = ""
            alert("Successfully uploaded");
        },
        onErrorItem: function(item, response, status, headers) {
            $scope.uploadError = "An error occurred while uploading file"
        }
    });
}]);