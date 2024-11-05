import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
require("@nomicfoundation/hardhat-verify");
require('dotenv').config();

const config: HardhatUserConfig = {
  solidity: "0.8.27",
  defaultNetwork: "polygon_amoy",
  networks: {
    hardhat: {
    },
    polygon_amoy: {
      url: process.env.POLYGON_RPC_URL,
      accounts: [process.env.PRIVATE_KEY as string]
    }
  },
  etherscan: {
    apiKey: process.env.POLYGONSCAN_API_KEY
  },
  sourcify: {
    enabled: true
  }
};

export default config;
