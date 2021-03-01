class Registry(object):
    pass


registry_machine = Registry('registry_machine')

registry_machine.register()
def print_hello_world(word):
    print('hello {}'.format(word))


registry_machine.register()
def print_hi_world(word):
    print('hi {}'.format(word))

if __name__ == '__main__':

    cfg1 = 'print_hello_word'
    registry_machine.get(cfg1)('world')

    cfg2 = 'print_hi_word'
    registry_machine.get(cfg2)('world')