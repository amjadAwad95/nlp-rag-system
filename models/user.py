import bcrypt
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    username: str = Field(...)
    email: EmailStr = Field(...)
    password: str = Field(...)



    @classmethod
    def create_user(cls, username: str, email: EmailStr, password: str):
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

        return cls(username=username, email=email, password=hashed)


class LoginUser(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)