
def user_individual_serial(user):
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
    }


def chat_individual_serial(chat):
    return {
        "id": str(chat["_id"]),
        "chat_type": chat["chat_type"],
        "documents": chat["documents"],
        "user_id": chat["user_id"],
        "chat_history": chat["chat_history"],
    }