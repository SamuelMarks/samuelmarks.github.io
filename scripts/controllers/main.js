'use strict';

var MainCtrl = angular.module('mainCtrl', []);

MainCtrl.controller('MainCtrl', function ($scope, $location) {
    $scope.go = function (path) {
        $location.path(path);
    };
});
