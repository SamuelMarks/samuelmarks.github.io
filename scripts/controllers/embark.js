'use strict';

var EmbarkCtrl = angular.module('embarkCtrl', []);

EmbarkCtrl.controller('EmbarkCtrl', function ($scope, $state, User) {
    $scope.previous_states = [];

    $scope.borrow = function (amount) {
        User.addDebt(amount);
    };

    $scope.next_state = function () {
        $state.go('embark.stories.enter');
    };
});
