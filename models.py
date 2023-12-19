from sqlalchemy import Column, Integer, String, Date, Time, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Aluno(Base):
    __tablename__ = 'aluno'

    prontuario = Column(String(9),primary_key=True,nullable=False)
    nome = Column(String(100))

    presenca = relationship('Presenca', back_populates='aluno')

    def __init__(self, prontuario,nome):
        self.prontuario = prontuario
        self.nome = nome

class Presenca(Base):
    __tablename__ = 'presenca'

    id = Column(Integer,primary_key=True)
    prontuario= Column(String(9), ForeignKey('aluno.prontuario'),primary_key=True)
    
    data = Column(Date)
    hora = Column(Time)
    
    aluno = relationship('Aluno', back_populates='presenca')

    def __init__(self, prontuario,data,hora):
        self.prontuario = prontuario
        self.data = data
        self.hora = hora
