data = open("branchless/Main.class", "rb").read()
print(data[0:4])

class Constant:
    entry_type: int
    entry_data: bytes
    
    @staticmethod
    def to_entry_type(data: int) -> str:
        return {
            1: "CONSTANT_Utf8",
            3: "CONSTANT_Integer", 
            4: "CONSTANT_Float",
            5: "CONSTANT_Long",
            6: "CONSTANT_Double",
            7: "CONSTANT_Class",
            8: "CONSTANT_String",
            9: "CONSTANT_Fieldref",
            10: "CONSTANT_Methodref",
            11: "CONSTANT_InterfaceMethodref",
            12: "CONSTANT_NameAndType",
            15: "CONSTANT_MethodHandle",
            16: "CONSTANT_MethodType",
            18: "CONSTANT_InvokeDynamic"
        }[data]

    def __init__(self, entry_type: int, entry_data: bytes):
        self.entry_type = entry_type
        self.entry_data = entry_data

    def __str__(self):
        return f"Constant(entry_type={Constant.to_entry_type(self.entry_type)}, entry_data={self.entry_data})"

class ConstantPool:
    constant_pool: list[Constant]

    def __init__(self, data: bytes):
        char_index = 0
        while char_index < len(data):
            entry_type = data[char_index]
            entry_data = data[char_index + 1:]
            self.constant_pool.append(Constant(entry_type, entry_data))
            char_index += len(entry_data) + 1

class ClassFile:
    magic: bytes
    minor_version: int
    major_version: int
    constant_pool_count: int

    def __init__(self, data: bytes):
        self.magic = data[0:4]
        self.minor_version = int.from_bytes(data[4:6], "big")
        self.major_version = int.from_bytes(data[6:8], "big")
        self.constant_pool_count = int.from_bytes(data[8:10], "big")
        self.constant_pool = data[10:self.constant_pool_count]
        
        self.access_flags = ClassFile.to_int(data[self.constant_pool_count:self.constant_pool_count + 2])
        self.this_class = ClassFile.to_int(data[self.constant_pool_count + 2:self.constant_pool_count + 4])
        self.super_class = ClassFile.to_int(data[self.constant_pool_count + 4:self.constant_pool_count + 6])
        self.interfaces_count = ClassFile.to_int(data[self.constant_pool_count + 6:self.constant_pool_count + 8])
        self.interfaces = data[self.constant_pool_count + 8:self.constant_pool_count + 8 + self.interfaces_count * 2]
        self.fields_count = ClassFile.to_int(data[self.constant_pool_count + 8 + self.interfaces_count * 2:self.constant_pool_count + 10 + self.interfaces_count * 2])
        self.fields = data[self.constant_pool_count + 10 + self.interfaces_count * 2:self.constant_pool_count + 10 + self.interfaces_count * 2 + self.fields_count * 8]

    def __str__(self):
        return f"ClassFile(magic={self.magic}, minor_version={self.minor_version}, major_version={self.major_version}, constant_pool_count={self.constant_pool_count})"

    @staticmethod
    def to_int(data: bytes) -> int:
        return int.from_bytes(data, "big")

class_file = ClassFile(data)
print(class_file)
