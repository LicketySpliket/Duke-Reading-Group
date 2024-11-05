const contractName = 'Token'
const contractAddress = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
const Contract = await ethers.getContractFactory(contractName)
const contract = await Contract.attach(contractAddress)
console.log(await contract.owner())

const acc0 = '0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266'
const acc0_key =
  '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80'

const acc1 = '0x70997970C51812dc3A010C7d01b50e0d17dc79C8'
const acc1_key =
  '0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d'

const wallet = new ethers.Wallet(acc0_key, ethers.provider)

const contractWithSigner = contract.connect(wallet)

const tx = await contractWithSigner.transfer(acc1)

// console.log(tx)

console.log(await contract.owner())
