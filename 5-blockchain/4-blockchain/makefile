.PHONY: setup build test up deploy console deployprod

setup:
	npm install

build:
	npx hardhat compile

test:
	npx hardhat test

up:
	npx hardhat node

deploy:
	npx hardhat ignition deploy ./ignition/modules/Token.ts --network localhost

console:
	npx hardhat console --network localhost

deployprod:
	npx hardhat ignition deploy ./ignition/modules/Token.ts --network polygon_amoy
# npx hardhat verify --network polygon_amoy 0x...