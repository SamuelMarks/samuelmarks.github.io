'use strict';

var UserCtrl = angular.module('userCtrl', []);

UserCtrl.controller('UserCtrl', function ($scope, User) {
    $scope.user = User.user();
});
