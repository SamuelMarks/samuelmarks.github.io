'use strict';

var EnterCtrl = angular.module('enterCtrl', []);

EnterCtrl.controller('EnterCtrl', function($scope, $modal, $log, User) {
    $log.info("Entered 'EnterCtrl'");
    var scopeVal = "Enter";
    if ($scope.previous_states.indexOf(scopeVal) === -1)
        $scope.previous_states.push(scopeVal);

    $scope.dice_roll = function() {
        var satoshi = Math.floor(Math.random() * 1000);
        if (Math.random() > 0.5) User.user().coins += satoshi;
        else User.user().coins -= satoshi;
    };

    $scope.items = ['item1', 'item2', 'item3'];

    $scope.open = function(size) {
        var modalInstance = $modal.open(
            {
                templateUrl: '/samuelmarks.github.io/views/story/exit.html',
                controller: ModalInstanceCtrl,
                size: size,
                resolve: {
                    items: function() {
                        return $scope.items;
                    }
                }
            });

        modalInstance.result.then(function(selectedItem) {
            $scope.selected = selectedItem;
        }, function() {
            $log.info('Modal dismissed at: ' + new Date());
        });
    };
});
