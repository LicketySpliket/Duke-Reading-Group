import { InjectedConnector } from '@web3-react/injected-connector'
import { ethers } from 'ethers'

export const injectedConnector = new InjectedConnector({
  supportedChainIds: [80002] // Amoy testnet chain ID
})

export const getProvider = () => {
  return new ethers.JsonRpcProvider('https://rpc-amoy.polygon.technology/')
}
