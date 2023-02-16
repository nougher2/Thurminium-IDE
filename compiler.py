# IMPORTS

import string as s

# CONSTANTS
TINT = "INT"
TFLO = "FLOAT"
TIDE = "IDENTIFIER"
TADD = "ADD"
TSUB = "SUB"
TMUL = "MULTIPLY"
TDIV = "DIVIDE"
TPOW = "POW"
TEQU = "EQUALS"
TLBR = "LBRACKET"
TRBR = "RBRACKET"
TCOM = "COMMA"
TEQO = "ET"
TNEO = "NET"
TLTO = "LT"
TGTO = "GT"
TLTE = "LTOE"
TGTE = "GTOE"
TKEY = "KEY"
TEOF = "EOF"

TNUM = "0123456789"
TLET = s.ascii_letters
TL_D = TNUM + TLET

keyword = ["AND", "OR", "NOT", "if", "elseif", "then", "else", "for", "in", "then", "while", "end"]

# ERROR HANDLERS
# Different types of errors for different types of stages and situations
class Error:
    # Defines Variables
    def __init__(self, posStart, posEnd, error, desc):
        self.posStart = posStart
        self.posEnd = posEnd
        self.error = error
        self.desc = desc

    # Creates report about the errors position and details for user
    def report(self):
        result = f"{self.error}: {self.desc}"
        result += "\n" + f"File {self.posStart.fn}, line {self.posStart.ln + 1}"
        result += "\n\n" + f"{self.posStart.ftxt}"
        return result

# Handles illegal characters in the text  
class charError(Error):
    def __init__(self, posStart, posEnd, desc):
        super().__init__(posStart, posEnd, "Illegal Character", desc)
        
class expectedCharError(Error):
    def __init__(self, posStart, posEnd, desc):
        super().__init__(posStart, posEnd, "Character Expected", desc)

# Handles syntax errors from the tokens
class syntaxError(Error):
    def __init__(self, posStart, posEnd, desc=''):
        super().__init__(posStart, posEnd, "Invalid Syntax", desc)

# Detects run time errors when interpreting from the Syntax tree
class runTimeError(Error):
    def __init__(self, posStart, posEnd, desc, trace):
        super().__init__(posStart, posEnd, "Run Time Error", desc)
        self.trace = trace
        
    def report(self):
        res = self.generateTrace()
        res += f"{self.error}: {self.desc}"
        res += "\n\n" + f"{self.posStart.ftxt}"
        return res
    
    # Traces back origin of where error was called from
    def generateTrace(self):
        res = ""
        pos = self.posStart
        trace = self.trace
        
        while trace:
            res += f"File {pos.fn}, line {str(pos.ln + 1)}, in {trace.displayName}\n"
            pos = trace.parentPos
            trace = trace.parent
        
        return "Traceback:\n" + res

# POSITON
# Stores positional data of tokens from text 

class Pos:
    # Defines variables for position
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    # Increments values of position in lexer
    def advance(self, curChar=None):
        self.idx += 1
        self.col += 1

        if curChar == "\n":
            self.ln += 1
            self.col = 0

        return self

    # Copys information of the current position
    def copy(self):
        return Pos(self.idx, self.ln, self.col, self.fn, self.ftxt)

# TOKENS
# Creates tokens with its identifier and value (if it has one)
class Tokens:
    # Defines variables
    def __init__(self, type, value=None, posStart=None, posEnd=None):
        self.type = type
        self.value = value
        
        # Checks for position start and copies the location if there is
        if posStart:
            self.posStart = posStart.copy()
            self.posEnd = posStart.copy()
            self.posEnd.advance()
        
        # Checks for position end and sets it to it if there is
        if posEnd:
            self.posEnd = posEnd

    # Function that returns token
    def __repr__(self):
        if self.value:
            return f"{self.type}:{self.value}"
        
        return f"{self.type}"

# LEXER
# Peforms lexical analysis on text
class Lexer:
    #Defines variables
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.posistion = Pos(-1, 0, -1, fn, text)
        self.curChar = None
        self.advance()

    # Increments to next character
    def advance(self):
        self.posistion.advance(self.curChar)

        # Determines if index has reached end of text and changes value depending on what it does
        if self.posistion.idx < len(self.text):
            self.curChar = self.text[self.posistion.idx]
        else:
            self.curChar = None

    # Identifies character and appends to token list
    def tokeniser(self):
        # Defines token list to store tokens
        toks = []

        # Determines current characters type and appends to list with value (if it has one)
        while self.curChar != None:
            if self.curChar == " ":
                pass
            elif self.curChar in TNUM:
                toks.append(self.numMaker())
            elif self.curChar in TLET:
                toks.append(self.idenMaker())
            elif self.curChar == ")":
                toks.append(Tokens(TRBR, posStart=self.posistion))
            elif self.curChar == "-":
                toks.append(Tokens(TSUB, posStart=self.posistion))
            elif self.curChar == "*":
                toks.append(Tokens(TMUL, posStart=self.posistion))
            elif self.curChar == "/":
                toks.append(Tokens(TDIV, posStart=self.posistion))
            elif self.curChar == "+":
                toks.append(Tokens(TADD, posStart=self.posistion))
            elif self.curChar == "(":
                toks.append(Tokens(TLBR, posStart=self.posistion))
            elif self.curChar == ",":
                toks.append(Tokens(TCOM, posStart=self.posistion))
            elif self.curChar == "^":
                toks.append(Tokens(TPOW, posStart=self.posistion))
            elif self.curChar == "=":
                toks.append(self.equalMaker())
            elif self.curChar == "!":
                tok, error = self.notEqualMaker()
                if error:
                    return [], error
                toks.append(tok)
            elif self.curChar == "<":
                toks.append(self.lessThanMaker())
            elif self.curChar == ">":
                toks.append(self.greaterThanMaker())
            else:
                # Stores position information for location of error aswell as details of error
                posStart = self.posistion.copy()
                char = f"'{self.curChar}'"
                return [], charError(posStart, self.posistion, char)

            self.advance()

        toks.append(Tokens(TEOF, posStart=self.posistion))
        # returns token table and no error if no were found
        return toks, None
    
    # Handles numbers with multiple characters and floating point
    def numMaker(self):
        # Defines variable
        temp = ""
        dcount = 0 
        posStart = self.posistion.copy()

        # Continually increments through list till it reachs end of text/number or finds a dot
        while self.curChar != None and self.curChar in TNUM + ".":
            if self.curChar == ".":
                if dcount == 1:
                    break
                dcount = 1
                temp += "."
            else:
                temp += self.curChar
            
            self.advance()

        # Determines if number is floating/integer
        if dcount == 1:
            return Tokens(TFLO, float(temp), posStart, self.posistion)
        else:
            return Tokens(TINT, int(temp), posStart, self.posistion)

    # Increments through letters till identifier made
    def idenMaker(self):
        # Creates temporary variable and stores current position of first letter
        temp = ""
        posStart = self.posistion.copy()
        
        while self.curChar != None and self.curChar in TL_D + "_":
            temp += self.curChar
            self.advance()
        
        tokType = TKEY if temp in keyword else TIDE
        return Tokens(tokType, temp, posStart, self.posistion)
    
    # Checks if logical operator != is made if not returns error
    def notEqualMaker(self):
        posStart = self.posistion.copy()
        self.advance()
        
        if self.curChar == "=":
            return Tokens(TNEO, posStart, self.posistion), None
        
        return None, expectedCharError(posStart, self.posistion, "Expected '=' after '!'")
        
    # Checks if logical operator is = or ==
    def equalMaker(self):
        tType = TEQU
        posStart = self.posistion.copy()
        self.advance()
        
        if self.curChar == "=":
            tType = TEQO
            
        return Tokens(tType, posStart, self.posistion)
    
    # Checks if logical operator is < or <=
    def lessThanMaker(self):
        tType = TLTO
        posStart = self.posistion.copy()
        self.advance()
        
        if self.curChar == "=":
            tType = TLTE
            
        return Tokens(tType, posStart, self.posistion)
    
    # Checks if logical operator is > or >=
    def greaterThanMaker(self):
        tType = TGTO
        posStart = self.posistion.copy()
        self.advance()
        
        if self.curChar == "=":
            tType = TGTE
            
        return Tokens(tType, posStart, self.posistion)
        
# NODES
# Different node types for syntax tree
class NumNode:
    # Defines variables
    def __init__(self, tok):
        self.tok = tok
        self.posStart = self.tok.posStart
        self.posEnd = self.tok.posEnd
    
    # Number node representation
    def __repr__(self):
        return f"{self.tok}"

class varAccessNode:
    def __init__(self, varTok):
        self.varTok = varTok

        self.posStart = self.varTok.posStart
        self.posEnd = self.varTok.posEnd

class varAssignNode:
    def __init__(self, varTok, varValue):
        self.varTok = varTok
        self.varValue = varValue

        self.posStart = self.varTok.posStart
        self.posEnd = self.varTok.posEnd

class OpNode:
    # Defines variables
    def __init__(self, LNode, OTok, RNode):
        self.LNode = LNode
        self.OTok = OTok
        self.RNode = RNode
        self.posStart = self.LNode.posStart
        self.posEnd = self.RNode.posEnd
    
    # Operation node representation
    def __repr__(self):
        return f"({self.LNode}, {self.OTok}, {self.RNode})"
        
class NegOpNode:
    # Defines variables
    def __init__(self, OTok, node):
        self.OTok = OTok
        self.node = node
        self.posStart = self.OTok.posStart
        self.posEnd = node.posEnd
    
    # Negative Num node representation
    def __repr__(self):
        return f"({self.OTok}, {self.node})"
        
class IfNode:
    def __init__(self, conds, elseCond):
        self.conds = conds
        self.elseCond = elseCond
        
        self.posStart = self.conds[0][0].posStart
        self.posEnd = (self.elseCond or self.conds[len(self.conds)-1][0]).posEnd
        
class ForNode:
    def __init__(self, varName, startNode, endNode, stepNode, exprNode):
        self.varName = varName
        self.startNode = startNode
        self.endNode = endNode
        self.stepNode = stepNode
        self.exprNode = exprNode
        
        self.posStart = self.varName.posStart
        self.posEnd = self.exprNode.posEnd
        
class WhileNode:
    def __init__(self, condNode, exprNode):
        self.condNode = condNode
        self.exprNode = exprNode
        
        self.posStart = self.condNode.posStart
        self.posEnd = self.condNode.posEnd

# PARSE RESULT
# Used to make error handling easier
class parseResult:
    # Defines variables
    def __init__(self):
        self.error = None
        self.node = None
        self.advanceCount = 0
        
    def registerAdvance(self):
        self.advanceCount += 1
        
    def registerUnadvance(self):
        self.advanceCount -= 1
    
    # How to represent parseResult
    def __repr__(self):
        return f"{self.error}, {self.node}"
    
    # Takes in result of a parse (factor/term/expression) and returns it as node
    def register(self, res):
        self.advanceCount += res.advanceCount
        if res.error:
            self.error = res.error
        return res.node

    # Sets res.node if it is successful
    def success(self, node):
        self.node = node
        return self
    
    # Sets res.error to a result if a parse fails
    def fail(self, error):
        if not self.error:
            self.error = error
        return self

# PARSER
class Parser:
    # Defines variables
    def __init__(self, tokens):
        self.tokens = tokens
        self.tIndex = -1
        self.advance()

    # Advances token index to next token to add to tree
    def advance(self):
        self.tIndex += 1

        if self.tIndex < len(self.tokens):
            self.curTok = self.tokens[self.tIndex]
        
        return self.curTok
        
    def unadvance(self):
        self.tIndex -= -1
        
        if self.tIndex > len(self.tokens):
            self.curTok = self.tokens[self.tIndex]
        
        return self.curTok
    
    # Starts the parse from the root
    def parse(self):
        res = self.express()
        # if no error and not EOF then there is still somethings left to parse so call error
        if not res.error and self.curTok.type != TEOF:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posEnd,
            "'+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', 'AND', 'OR' expected"
            ))
        return res
    
    # Checks if token meets factor requirements
    def factor(self):
        # Defines class and current token
        res = parseResult()
        tok = self.curTok
        
        # Checks if factor is negative number and stores is as neg number token         
        if tok.type in [TADD, TSUB]:
            res.registerAdvance()
            self.advance()
            factor = res.register(self.factor())
            
            if res.error:
                return res
                
            return res.success(NegOpNode(tok, factor))
        # Checks if factor is integer or float
        elif tok.type in [TINT, TFLO]:
            res.registerAdvance()
            self.advance()
            return res.success(NumNode(tok))
        
        # Checks if it is a variable
        elif tok.type == TIDE:
            res.registerAdvance()
            self.advance()
            return res.success(varAccessNode(tok))
        
        # Checks if factor is a pair of brackets
        elif tok.type == TLBR:
            # Collects rest of expression
            res.registerAdvance()
            self.advance()
            exp = res.register(self.express())
            
            if res.error:
                return res
            
            # Checks if expression ends with bracket returns error if not
            if self.curTok.type == TRBR:
                res.registerAdvance()
                self.advance()
                return res.success(exp)
            else:
                return res.fail(syntaxError(tok.posStart, tok.posEnd, "')' is expected"))
        
        elif tok.type in TKEY and tok.value == "if":
            ifExpress = res.register(self.ifExpress())
            if res.error:
                return res
            return res.success(ifExpress)
            
        elif tok.type in TKEY and tok.value == "for":
            forExpress = res.register(self.forExpress())
            if res.error:
                return res
            return res.success(forExpress)
            
        elif tok.type in TKEY and tok.value == "while":
            whileExpress = res.register(self.whileExpress())
            if res.error:
                return res
            return res.success(whileExpress)
        
        # If Int or float is not detected return syntax error
        return res.fail(syntaxError(tok.posStart, tok.posEnd, "Int, float or identifier expected"))
    
    # Sends information for term to operate
    def term(self):
        return self.operation(self.factor, [TDIV, TMUL, TPOW])
        
    def arithExpress(self):
        return self.operation(self.term, [TADD, TSUB])
    
    def compExpress(self):
        res = parseResult()
        
        if self.curTok.type == TKEY and self.curTok.value == "NOT":
            OTok = self.curTok
            res.registerAdvance()
            self.advance()
            
            node = res.register(self.compExpress())
            
            if res.error:
                return res
                
            return res.success(NegOpNode(OTok, node))
        
        print("arith express")
        ops = [TEQO, TNEO, TLTO, TGTO, TLTE, TGTE]
        node = res.register(self.operation(self.arithExpress, ops))
        
        if res.error:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posEnd, "Expected int, float, identifier, '+', '-', ')', 'NOT'"))
            
        return res.success(node)
        
    def ifExpress(self):
        res = parseResult()
        conds = []
        elseCond = None

        if not self.curTok.type in TKEY and self.curTok.value == "if":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posEnd, "Expected 'if' statement"))

        res.registerAdvance()
        self.advance()

        condition = res.register(self.express())
		
        if res.error: 
            return res

        if not self.curTok.type in TKEY and self.curTok.value == "then":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posEnd, "Expected 'then' after 'if'"))

        res.registerAdvance()
        self.advance()

        express = res.register(self.express())
		
        if res.error: 
            return res
		
        conds.append((condition, express))

        while self.curTok.type in TKEY and self.curTok.value == "elseif":
            res.registerAdvance()
            self.advance()

            condition = res.register(self.express())
            if res.error: 
                return res

            if not self.curTok.type in TKEY and self.curTok.value == "then":
                return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'then' after 'elseif'"))

            res.registerAdvance()
            self.advance()

            express = res.register(self.express())
            if res.error: 
                return res
            conds.append((condition, express))

        if self.curTok.type in TKEY and self.curTok.value == "else":
            res.registerAdvance()
            self.advance()

            elseCond = res.register(self.express())
            if res.error: 
                return res
                
        res.registerAdvance()
        self.advance()
                
        if not self.curTok.type in TKEY and self.curTok.value == "end":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'end' at end 'if' statement"))
        
        return res.success(IfNode(conds, elseCond))
        
    def forExpress(self):
        res = parseResult()
        
        if not self.curTok.type in TKEY and self.curTok.value == "for":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'for' statement"))
            
        res.registerAdvance()
        self.advance()
        
        if self.curTok.type != TIDE:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected identifier"))
            
        varName = self.curTok
        res.registerAdvance()
        self.advance()
        
        if not self.curTok.type in TKEY and self.curTok.value == "in":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'in'"))
        
        res.registerAdvance()
        self.advance()
        
        if self.curTok.type != TLBR:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected '('"))
        
        res.registerAdvance()
        self.advance()
        
        startValue = res.register(self.express())
        if res.error:
            return res
            
        print(self.curTok.type)
        print(self.curTok.value)
            
        if self.curTok.type != TCOM:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected ','"))
            
        res.registerAdvance()
        self.advance()
        
        endValue = res.register(self.express())
        if res.error:
            return res
            
        if self.curTok.type == TCOM:
            res.registerAdvance()
            self.advance()
            
            stepValue = res.register(self.express())
            if res.error:
                return res
        else:
            stepValue = None
            
        res.registerAdvance()
        self.advance()
        
        print(self.curTok.type)
        print(self.curTok.value)
        
        if self.curTok.type != TRBR:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected ')'"))
            
        res.registerAdvance()
        self.advance()
        
        if not self.curTok.type in TKEY and self.curTok.value == "then":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'then'"))
            
        res.registerAdvance()
        self.advance()
        
        body = res.register(self.express())
        if res.error:
            return res
            
        return res.success(ForNode(varName, startValue, endValue, stepValue, body))
        
    def whileExpress(self):
        res = parseResult()
        
        if not self.curTok.type in TKEY and self.curTok.value == "while":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'while' statement"))
            
        res.registerAdvance()
        self.advance()
        
        condition = res.register(self.express())
        if res.error:
            return res
            
        if not self.curTok.type in TKEY and self.curTok.value == "then":
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posStart, "Expected 'while' statement"))
            
        res.registerAdvance()
        self.advance()
        
        body = res.register(self.express())
        if res.error:
            return res
            
        return res.success(WhileNode(condition, body))

    # Sends information for expression to operate
    def express(self):
        res = parseResult()

        if self.curTok.type == TIDE:
            varName = self.curTok
            res.registerAdvance()
            self.advance()
            
            if self.curTok.type in [TEQO, TNEO, TLTO, TGTO, TLTE, TGTE]:
                OTok = self.curTok
                res.registerAdvance()
                self.advance()
                right = res.register(self.factor())
                
                if res.error:
                    return res
                    
                return res.success(OpNode(varAccessNode(varName), OTok, right))

            if self.curTok.type != TEQU:
                
                if varName.value in _GST.symbols.keys():
                    if self.curTok.type in [TADD, TSUB, TMUL, TDIV, TPOW]:
                        OTok = self.curTok
                        res.registerAdvance()
                        self.advance()
                        right = res.register(self.factor())
                        
                        if res.error:
                            return res
                            
                        return res.success(OpNode(varAccessNode(varName), OTok, right))
                    else:
                        return res.success(varAccessNode(varName))
                else:
                    return res.fail(syntaxError(self.curTok.posStart, self.curTok.posEnd,
                    "Expected '='"))

            res.registerAdvance()
            self.advance()
            express = res.register(self.express())

            if res.error:
                return res
            
            return res.success(varAssignNode(varName, express))
            
        node = res.register(self.operation(self.compExpress, ((TKEY, "AND"), (TKEY, "OR"))))
        
        if res.error:
            return res.fail(syntaxError(self.curTok.posStart, self.curTok.posEnd,
            "Expected int, float, identifier, '+', '-' or '('"))
        
        print(self.tokens)
        return res.success(node)
    
    # Checks if term/expression meet their grammar requirements
    def operation(self, func, ops):
        # Defines parse result and calls function
        res = parseResult()
        left = res.register(func())
        
        if res.error:
            return res
        
        # If token is operator stored in variable and the next factor is stored in right node
        while self.curTok.type in ops or (self.curTok.type, self.curTok.value) in ops:
            OTok = self.curTok
            print(OTok)
            res.registerAdvance()
            self.advance()
            right = res.register(func())
            
            if res.error:
                return res
            
            # Creates Operation node for factor ^ operation ^ factor
            left = OpNode(left, OTok, right)
        
        # returns term/expression as res.node
        return res.success(left)
        
class runTimeRes:
    def __init__(self):
        self.value = None
        self.error = None
        
    def reg(self, res):
        if res.error:
            self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self
        
    def fail(self, error):
        self.error = error
        return self
        
# VALUES
# Stores numbers to be operated on with other numbers
class Num:
    # Defines variables and methods
    def __init__(self, value):
        self.value = value
        self.setPos()
        self.setTrace()
    
    # Stores position of number for error reporting
    def setPos(self, posStart=None, posEnd=None):
        self.posStart = posStart
        self.posEnd = posEnd
        return self
    
    def setTrace(self, trace=None):
        self.trace = trace
        return self
    
    # Addition operation
    def addition(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value + oNum.value).setTrace(self.trace), None
    
    # Subtraction operation
    def subtraction(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value - oNum.value).setTrace(self.trace), None
    
    # Multiplication operation
    def multiplication(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value * oNum.value).setTrace(self.trace), None
    
    # Division operation
    def division(self, oNum):
        if isinstance(oNum, Num):
            if oNum.value == 0:
                return None, runTimeError(oNum.posStart, oNum.posEnd, "Cannot Divide by 0", self.trace)
            
            return Num(self.value / oNum.value).setTrace(self.trace), None
    
    # Power operation
    def power(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value ** oNum.value).setTrace(self.trace), None

    # Equals to operation     
    def comparisonEQ(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value == oNum.value).setTrace(self.trace), None

    # Not equals to operation
    def comparisonNE(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value != oNum.value).setTrace(self.trace), None

    # Less than operation
    def comparisonLT(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value < oNum.value).setTrace(self.trace), None

    # Greater than operation
    def comparisonGT(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value > oNum.value).setTrace(self.trace), None

    # Less than equal to operation
    def comparisonLTE(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value <= oNum.value).setTrace(self.trace), None

    # Greater than equal to operation
    def comparisonGTE(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value >= oNum.value).setTrace(self.trace), None

    # And operation
    def getAND(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value and oNum.value).setTrace(self.trace), None

    # Or operation
    def getOR(self, oNum):
        if isinstance(oNum, Num):
            return Num(self.value or oNum.value).setTrace(self.trace), None

    # Not operation    
    def getNOT(self):
        return Num(True if self.value == False else False).setTrace(self.trace), None

    # Copys the trace of the code
    def copy(self):
        copy = Num(self.value)
        copy.setPos(self.posStart, self.posEnd)
        copy.setTrace(self.trace)
        return copy
        
    def isTrue(self):
        return self.value != False
    
    # Representation method for number
    def __repr__(self):
        return str(self.value)

class Trace:
    def __init__(self, displayName, parent=None, parentPos=None):
        self.displayName = displayName
        self.parent = parent
        self.parentPos = parentPos
        self.SymbolTable = None
        
class symbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
        
    def Get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value
        
    def Set(self, name, value):
        self.symbols[name] = value
        
    def remove(self, name):
        del self.symbols[name]
    

# INTERPRETER
# Traverses syntax tree and executes in logical order 
class interpreter:
    # Takes node, processes and visit its child nodes (if it has one)
    def visit(self, node, trace):
        methodName = f"v{type(node).__name__}"
        # follows method of node and if not found defaults to no method
        method = getattr(self, methodName, self.noMethod)
        return method(node, trace)
        
    def noMethod(self, node, trace):
        raise Exception(f"No visit.{type(node).__name__} defined")
    
    # Visits number node
    def vNumNode(self, node, trace):
        return runTimeRes().success(Num(node.tok.value).setTrace(trace).setPos(node.posStart, node.posEnd))
    
    def vvarAccessNode(self, node, trace):
        res = runTimeRes()
        varName = node.varTok.value
        value = trace.SymbolTable.Get(varName)
        
        if not value:
            return res.fail(runTimeError(node.posStart, node.posEnd, f"'{varName}' not defined", trace))
            
        value = value.copy().setPos(node.posStart, node.posEnd)
        return res.success(value)
        
    def vvarAssignNode(self, node, trace):
        res = runTimeRes()
        varName = node.varTok.value
        value = res.reg(self.visit(node.varValue, trace))
        
        if res.error:
            return res
            
        trace.SymbolTable.Set(varName, value)
        return res.success(value)
    
    # Visits Operation node and child nodes
    def vOpNode(self, node, trace):
        print("Operation node found")
        # Defines runtime result class and left and right node
        res = runTimeRes()
        left = res.reg(self.visit(node.LNode, trace))
        
        if res.error:
            return res
        
        right = res.reg(self.visit(node.RNode, trace))
        
        if res.error:
            return res
        
        # Depending on operation it will execute it on 2 child nodes
        if node.OTok.type == TADD:
            result, error = left.addition(right)
        elif node.OTok.type == TSUB:
            result, error = left.subtraction(right)
        elif node.OTok.type == TMUL:
            result, error = left.multiplication(right)
        elif node.OTok.type == TDIV:
            result, error = left.division(right)
        elif node.OTok.type == TPOW:
            result, error = left.power(right)
        elif node.OTok.type == TEQO:
            result, error = left.comparisonEQ(right)
        elif node.OTok.type == TNEO:
            result, error = left.comparisonNE(right)
        elif node.OTok.type == TLTO:
            result, error = left.comparisonLT(right)
        elif node.OTok.type == TGTO:
            result, error = left.comparisonGT(right)
        elif node.OTok.type == TLTE:
            result, error = left.comparisonLTE(right)
        elif node.OTok.type == TGTE:
            result, error = left.comparisonGTE(right)
        elif node.OTok.type == TKEY and node.OTok.value == "AND":
            result, error = left.getAND(right)
        elif node.OTok.type == TKEY and node.OTok.value == "OR":
            result, error = left.getOR(right)
        
        if error:
            return res.fail(error)
        else:
            return res.success(result.setPos(node.posStart, node.posEnd))
    
    # Visits Negative operation node
    def vNegOpNode(self, node, trace):
        res = runTimeRes()
        nums = res.reg(self.visit(node.node, trace))
        
        if res.error:
            return res
            
        error = None
        
        if node.OTok.type == TSUB:
            nums, error = nums.multiplication(Num(-1))
        if node.OTok.type == TKEY and node.OTok.value == "NOT":
            nums, error = nums.getNOT()
            
        if error:
            return res.fail(error)
        else:
            return res.success(nums.setPos(node.posStart, node.posEnd))
            
    def vIfNode(self, node, trace):
        res = runTimeRes()
        
        for condition, expr in node.conds:
            condValue = res.reg(self.visit(condition, trace))
            if res.error:
                return res
                
            if condValue.isTrue():
                exprValue = res.reg(self.visit(expr, trace))
                if res.error:
                    return res
                return res.success(exprValue)
                
        if node.elseCond:
            elseValue = res.reg(self.visit(node.elseCond, trace))
            if res.error:
                return res
            return res.success(elseValue)
            
        return res.success(None)
        
    def vForNode(self, node, trace):
        res = runTimeRes()
        
        startValue = res.reg(self.visit(node.startNode, trace))
        if res.error:
            return res
            
        endValue = res.reg(self.visit(node.endNode, trace))
        if res.error:
            return res
            
        if node.stepNode:
            stepValue = res.reg(self.visit(node.stepNode, trace))
            if res.error:
                return res
        else:
            stepValue = Num(1)
            
        i = startValue.value
        
        if stepValue.value >= 0:
            condition = lambda: i < endValue.value
        else:
            condition = lambda: i > endValue.value
            
        while condition():
            trace.symbolTable.Set(node.varName.value, Num(i))
            i += stepValue.value
            
            res.reg(self.visit(node.exprNode, trace))
            if res.error:
                return res
                
        return res.success(None)
        
        def vWhileNode(self, node, trace):
            res = runTimeRes()
            
            while True:
                condition = res.reg(self.visit(node.condNode, trace))
                if res.error:
                    return res
                    
                if not condition.isTrue():
                    break
                
                res.reg(self.visit(node.exprNode, trace))
                if res.error:
                    return res
                    
            return res.success(None)
        
        
        

_GST = symbolTable()
_GST.Set("nil", Num(0))

print(_GST.symbols)

    
# Used to go through each stage of compilation and pass result to next stage
def run(fn, text):
    # Lexical Analysis
    # Creates tokens from text and stores file name
    lexer = Lexer(fn, text)
    tokens, error = lexer.tokeniser()
    
    print(tokens)

    if error:
        return None, error
    
    # Syntax Analysis
    # Creates parse tree from tokens (from previous stage)
    parser = Parser(tokens)
    st = parser.parse()
    
    if st.error:
        return None, st.error
    
    # Semantic Analysis
    # Takes parse tree and executes in logical order of the tree
    Interpreter = interpreter()
    trace = Trace('<program>')
    trace.SymbolTable = _GST
    result = Interpreter.visit(st.node, trace)
    
    # Returns value of compilation
    return result.value, result.error
