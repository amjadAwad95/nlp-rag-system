import bcrypt
from fastapi import APIRouter, Path, HTTPException, Query
from database import user_collection
from schema import user_individual_serial
from models import User, LoginUser
from database import user_collection

user_router = APIRouter()

@user_router.get("/users")
async def get_users(
        email: str = Query(default=None, regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        username: str = Query(default=None),
):
    if email:
        user = user_collection.find_one({"email": email})

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return user_individual_serial(user)

    if username:
        user = user_collection.find_one({"username": username})

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return user_individual_serial(user)

    users = user_collection.find()

    return [user_individual_serial(user) for user in users]


@user_router.post("/users")
async def create_user(user: User):
    user_model = User.create_user(username=user.username, email=user.email, password=user.password)

    user_dict = user_model.model_dump()

    inserted_user = user_collection.insert_one(user_dict)

    user_dict.update({"_id": inserted_user.inserted_id})

    return user_individual_serial(user_dict)


@user_router.post("/users/login")
async def login(login_user: LoginUser):
    user = user_collection.find_one({"email": login_user.email})

    if not user:
        raise HTTPException(status_code=404, detail="Email or password incorrect")

    is_match = bcrypt.checkpw(login_user.password.encode("utf-8"), user["password"].encode("utf-8"))

    if not is_match:
        raise HTTPException(status_code=404, detail="Email or password incorrect")

    return user_individual_serial(user)
