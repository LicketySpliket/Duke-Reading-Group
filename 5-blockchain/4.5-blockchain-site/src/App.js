import logo from './logo.svg'
import './App.css'

import React, { useState, useEffect } from 'react'
import Web3 from 'web3'
import TokenContract from './TokenContract.json'
import { ethers } from 'ethers'

const web3 = new Web3('https://rpc-amoy.polygon.technology/')
const CONTRACT_ADDRESS = '0x440A426E44500D9a1c7E00869532b24E910DE68c'
const CONTRACT_ABI = TokenContract.abi

function App () {
  const [owner, setOwner] = useState(null)
  const [account, setAccount] = useState(null)
  const [signer, setSigner] = useState(null)

  const loadBlockchainData = async () => {
    let contract = new web3.eth.Contract(CONTRACT_ABI, CONTRACT_ADDRESS)

    const owner = await contract.methods.owner().call()
    setOwner(owner)

    const provider = new ethers.BrowserProvider(window.ethereum)
    const signer = await provider.getSigner()
    const account = await signer.getAddress()
    setAccount(account)
    setSigner(signer)
  }

  const transferOwnership = async () => {
    const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer)
    const tx = await contract.transfer(
      '0x0000000000000000000000000000000000000000'
    )
    await tx.wait()

    const owner = await contract.owner()
    setOwner(owner)
  }

  useEffect(() => {
    loadBlockchainData()
  }, [])

  return (
    <p>
      Owner: {owner}
      <br />
      Account: {account}
      <br />
      <button onClick={transferOwnership}>Transfer Ownership</button>
    </p>
  )
}

export default App
