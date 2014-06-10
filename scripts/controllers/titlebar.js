'use strict';

var TitleBarCtrl = angular.module('titlebarCtrl', []);

TitleBarCtrl.controller('TitleBarCtrl', function ($scope, $location) {
    $scope.isActive = function (route) {
        return route === $location.path();
    };
});
