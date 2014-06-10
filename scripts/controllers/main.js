'use strict';

var MainCtrl = angular.module('mainCtrl', []);

MainCtrl.controller('MainCtrl', function ($scope) {
    $scope.stuff = [1, 2, 3, 'a', 'b', 'c'];
});
