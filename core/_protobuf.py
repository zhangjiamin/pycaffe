
f = open('a.txt', 'r')
address_book = addressbook_pb2.AddressBook() # replace with your own message
text_format.Parse(f.read(), address_book)
f.close()

f = open('b.txt', 'w')
f.write(text_format.MessageToString(address_book))
f.close()


