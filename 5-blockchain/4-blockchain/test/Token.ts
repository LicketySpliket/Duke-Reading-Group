import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox/network-helpers";
import { anyValue } from "@nomicfoundation/hardhat-chai-matchers/withArgs";
import { expect } from "chai";
import hre from "hardhat";

describe("Token", function () {
    async function deployTokenFixture() {
        const [owner, otherAccount] = await hre.ethers.getSigners();

        const Token = await hre.ethers.getContractFactory("Token");
        const token = await Token.deploy();
        return { token, owner, otherAccount };
    }

    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            const { token, owner } = await loadFixture(deployTokenFixture);

            expect(await token.owner()).to.equal(owner.address);
        });
    });

    describe("Transfer", function () {
        it("Should transfer tokens", async function () {
            const { token, owner, otherAccount } = await loadFixture(deployTokenFixture);

            await token.transfer(otherAccount.address);

            expect(await token.owner()).to.equal(otherAccount.address);
        });
    });
});