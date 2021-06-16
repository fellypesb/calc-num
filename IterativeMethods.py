#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:26:48 2021

@author: Fellype Siqueira Barroso <fellypecsiqueira@gmail.com>
"""

import numpy as np

class IterativeMethods:
    '''
    Esta classe implementa 4 métodos iterativos que podem ser 
    utilizados para encontrar zeros de funções. Para realizar o estudo
    comparativo desses métodos, intancie esta classe fornecendo seus parâmetros 
    comuns, e para cada método, forneça seus parâmetros particulares.
    
    Parâmetros
    --------------
    f: lambda
        Função a qual irá ser utilizada em todos os métodos com 
        objetivo de encontrar suas raízes.

    tol: float
        Tolerância para critério de parada dos métodos.
        
    maxiter: int
        Número máximo de iterações desejadas.

    round_: int
         Número de casas decimais de precisão desejada.
    
    Exemplos
    ----------------
    
    >>> from IterativeMethods import IterativeMethods
    
    >>> f = lambda x : x**3-9*x+3
    >>> df = lambda x:3*x**2-9

    >>> methods = IterativeMethods(f=f, tol=5e-4, maxiter=20, round_=5)

    >>> bisseccao = methods.bisseccao(a=0, b=1)
    >>> newton = methods.newton(x=0.5, df=df)
    >>> secante = methods.secant(x1=0, x2=1)
         
    '''
    
    def __init__(self, f, tol:float, maxiter:int, round_:int):
        self.f = f
        self.tol = tol
        self.maxiter = maxiter
        self.round_ = round_
        
    def bisseccao(self, a, b, verbose=False):
        '''
            Função que calcula o zero de funções com o método da Bissecção.
            
            O processo iterativo deste método é realizado da seguinte forma:
            
            x = (a + b)/2

            Parâmetros
            --------------
            a: float
                Extremo esquerdo do intervalo de inspeção [a,b].

            b: float
                Extremo direito do intervalo de inspeção [a,b].

            verbose: bool, default=False
                Se True, o progresso das iterações é impresso na tela.

            Saída
            -----------
            dict contendo a quantidade de iterações necessárias para convergência
            e valor x, tal que f(x) é aprox. 0.
        '''
        
        if self.f(a)*self.f(b) > 0:
            raise ValueError("Não há raiz no intervalo fornecido")

        iter_ = 0

        while(iter_ <= self.maxiter and abs(b - a) > self.tol):
            x = round((a + b)/2, self.round_)      

            f_a = self.f(a)
            f_b = self.f(b)
            f_x = self.f(x)

            if(f_a * f_x > 0):
                a = x
            else:
                b = x

            iter_ += 1
            if verbose:
                print(f'Iteração: {iter_} \t Aproximação: {x}')
        if iter_ > self.maxiter:
            print("Número máximo de iterações alcançado!")
        else:
            return {'iter':iter_, 'raiz': x}
    
    def fixedpoint(self, x, g, verbose=False):
        '''
            Função que calcula o zero de funções com o método do Ponto Fixo.
            O processo iterativo é realizado da seguinte forma:

            x = g(x)

            Parâmetros
            --------------
            x: float
                Chute inicial.

            g: lambda
                Função alcançada apartir de f(x).
                Ex: f(x) = xe^x-1
                    g(x) = 1/e^x

            verbose: bool, default=False
                Se True, o progresso das iterações é impresso na tela.

            Saída
            -----------
            dict contendo a quantidade de iterações necessárias para convergência
            e valor x, tal que f(x) é aprox. 0.
        '''
        
        iter_ = 0 
        xj = x # anterior
        xi = x # atual
        err = np.inf
        while( iter_ <= self.maxiter and abs(err) > self.tol):
            xi = round(g(xj), self.round_)
            err = (xi-xj)
            xj = xi
            iter_ += 1
            if verbose:
                print(f'Iterações: {iter_} \t Aproximação: {xi}') 
        if iter_ > self.maxiter:
            print("Número máximo de iterações alcançado!")
        else:
            return {'iter':iter_, 'raiz': xi}
        
    def newton(self, x, df, verbose=False):
        '''
            Função que calcula o zero de funções com o método de Newton.
            O processo iterativo é realizado da seguinte forma:

            x = x - f(x)/f'(x)

            Parâmetros
            --------------
            x: float
                Chute inicial.

            df: lambda
                Derivada da função a qual queremos encontrar suas raízes

            verbose: bool, default=False
                Se True, o progresso das iterações é impresso na tela.

            Saída
            -----------
            dict contendo a quantidade de iterações necessárias para convergência
            e valor x, tal que f(x) é aprox. 0.
        '''
        
        iter_ = 0 
        while( iter_ <= self.maxiter and abs(self.f(x)) > self.tol):

            x = round(x - (self.f(x)/df(x)), self.round_)
            iter_ += 1
            if verbose:
                 print(f'Iterações: {iter_} \t Aproximação: {x}') 
        if iter_ > self.maxiter:
            print("Número máximo de iterações alcançado!")
        else:
            return {'iter':iter_, 'raiz': x}
        
    def secant(self, x1, x2, verbose=False):
        '''
            Função que calcula o zero de funções com o método da Secante.
            
            O método iterativo é realizado da Seguinte forma:

            x2 = (x1*f(x2) - x2*f(x1)) / (f(x2) - f(x1))  

            Parametros
            --------------
            x1: float
                Primeiro chute inicial.

            x2: float
                Segundo chute inicial.

            verbose: bool, default=False
                Se True, o progresso das iterações é impresso na tela.

            Saída
            -----------
            dict contendo a quantidade de iterações necessárias para convergência
            e valor x, tal que f(x) é aprox. 0.
        '''
        iter_ = 0

        while( iter_ <= self.maxiter and abs(self.f(x2)) > self.tol):
            c = x2
            x2 = round((x1*self.f(x2) - x2*self.f(x1)) / (self.f(x2) - self.f(x1)), self.round_)
            x1 = c
            iter_ += 1
            if verbose:
                print(f'Iterações: {iter_} \t Raiz: {x2}')

        if iter_ > self.maxiter:
            print("Número máximo de iterações alcançado!")
        else:
            return {'iter':iter_, 'raiz': x2}
