people = ["Dr jet Brook","Dr lee Hoo"]
def split_name(name):
    return  name.split()[0] +name.split()[-1]
for person in people:
    print(split_name(person) == (lambda x: (x.split()[0] +x.split()[-1]))(person))
    #print((lambda x: (x.split()[0] +x.split()[-1]))(person))
    #print(lambda x: (x.split()[0] +x.split()[-1])(person))