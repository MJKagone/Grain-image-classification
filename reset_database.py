import database as db

# Script for resetting database, use with caution

while True:
    print("Write CANCEL to exit.")
    user_input = input("Write CONFIRM to reset database: ")

    if user_input == "CONFIRM":
        db.reset()
        break
    elif user_input == "CANCEL":
        break
