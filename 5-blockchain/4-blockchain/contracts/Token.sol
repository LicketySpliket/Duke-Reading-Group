// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.27;

// represents one token

contract Token {
    address public owner;

    event Transfer(address indexed from, address indexed to);

    constructor() {
        owner = msg.sender;
    }

    function transfer(address to) public {
        require(msg.sender == owner, "You aren't the owner");
        emit Transfer(owner, to);
        owner = to;
    }
}