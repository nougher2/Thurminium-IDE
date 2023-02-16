import temp as M

while True:
    text = input("Thurminium > ")
    result, error = M.run("<Thurman-File>", text)

    if error:
        print(error.report())
    elif result:
        print(result)