
class NodeClass:
    def __init__(self, value):
        self.node = {
            'value': value,
            'next': None
        }

    def new_node(self):
        return self.node

class LinkedList:
    def __init__(self, value):
        self.head = {
            'value': value,
            'next': None
        }
        self.tail = self.head
        self.length = 1

    def append(self, value):
        """append a node to the end of the linked list"""
        new_node = NodeClass(value).new_node()
        self.tail['next'] = new_node
        self.tail = new_node
        self.length += 1

    def prepend(self, value):
        """ prepend a node to the head of the linked list"""
        new_node = NodeClass(value).new_node()
        new_node['next'] = self.head
        self.head = new_node
        self.length += 1

    def print_list(self):
        """ print the list as an array"""
        arr = []
        current_node = self.head
        while current_node:
            arr.append(current_node['value'])
            current_node = current_node['next']
        print(arr)

    def insert(self, value, index):
        """ insert a value between two nodes"""
        if index >= self.length:
            self.append(value)
        if index <= 0:
            self.prepend(value)
        new_node = NodeClass(value).new_node()
        leader = self.traverse_to_leader(index - 1)
        holding_pointer = leader['next']
        leader['next'] = new_node
        new_node['next'] = holding_pointer
        self.length += 1
        return self.print_list()

    def traverse_to_leader(self, index):
        if index < 0:
            raise ValueError("index cannot be negative")
        counter = 0
        current_node = self.head
        while counter != index:
            current_node = current_node['next']
            counter += 1
        return current_node

    def traverse_to_tail(self):
        current_node = self.head
        while current_node['next']['next']:
            current_node = current_node['next']
        return current_node

    def remove(self, index):
        if index == 0:
            head = self.head
            new_head = self.head['next']
            del head
            self.head = new_head
            self.length -= 1
        elif index > self.length:
            tail = self.traverse_to_tail()
            tail['next'] = None
             self.length -= 1




        else:
            leader = self.traverse_to_leader(index - 1)
            remove_point = leader['next']
            next_point = remove_point['next']
            leader['next'] = next_point