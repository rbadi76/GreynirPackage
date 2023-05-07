/*

   Greynir: Natural language processing for Icelandic

   C++ Earley parser module

   Copyright (C) 2022 Miðeind ehf.

   This software is licensed under the MIT License:

      Permission is hereby granted, free of charge, to any person
      obtaining a copy of this software and associated documentation
      files (the "Software"), to deal in the Software without restriction,
      including without limitation the rights to use, copy, modify, merge,
      publish, distribute, sublicense, and/or sell copies of the Software,
      and to permit persons to whom the Software is furnished to do so,
      subject to the following conditions:

      The above copyright notice and this permission notice shall be
      included in all copies or substantial portions of the Software.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
      EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
      MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
      IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
      CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
      TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
      SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   This module implements an optimized Earley parser in C++.
   It is designed to be called from Python code with
   already parsed and packed grammar structures.

   The Earley parser used here is the improved version described by Scott & Johnstone,
   referencing Tomita. This allows worst-case cubic (O(n^3)) order, where n is the
   length of the input sentence, while still returning all possible parse trees
   for an ambiguous grammar.

   See Elizabeth Scott, Adrian Johnstone:
   "Recognition is not parsing — SPPF-style parsing from cubic recognisers"
   Science of Computer Programming, Volume 75, Issues 1–2, 1 January 2010, Pages 55–70

*/

#include <stdlib.h>
#include <string.h>
#include <wchar.h>


// Assert macro
#ifdef DEBUG
   #define ASSERT(x) assert(x)
#else
   #define ASSERT(x)
#endif


typedef unsigned int UINT;
typedef int INT;
typedef wchar_t WCHAR;
typedef char CHAR;
typedef unsigned char BYTE;
typedef bool BOOL;


class Production;
class Parser;
class State;
class Column;
class NodeDict;
class NodeDict2;
class TauOrPsiDict;
class Label;
struct StateChunk;


class AllocCounter {

   // A utility class to count allocated instances
   // of an instrumented class. Add this as a static
   // member (named e.g. 'ac') of the class to be watched
   // and call ac++ and ac-- in the constructor and destructor,
   // respectively.

private:

   UINT m_nAllocs;
   UINT m_nFrees;

public:

   AllocCounter(void)
      : m_nAllocs(0), m_nFrees(0)
      { }
   ~AllocCounter(void)
      { }

   void operator++(int)
      { this->m_nAllocs++; }
   void operator--(int)
      {
         ASSERT(this->m_nAllocs > this->m_nFrees);
         this->m_nFrees++;
      }
   UINT numAllocs(void) const
      { return this->m_nAllocs; }
   UINT numFrees(void) const
      { return this->m_nFrees; }
   INT getBalance(void) const
      { return (INT)(this->m_nAllocs - this->m_nFrees); }

};


class Nonterminal {

   // A Nonterminal has an associated list of owned Productions

   friend class AllocReporter;

private:

   WCHAR* m_pwzName;
   Production* m_pProd;

   static AllocCounter ac;

protected:

public:

   Nonterminal(const WCHAR* pwzName);

   ~Nonterminal(void);

   void addProduction(Production* p);

   // Get the first right-hand-side production of this nonterminal
   Production* getHead(void) const
      { return this->m_pProd; }

   WCHAR* getName(void) const
      { return this->m_pwzName; }

};


class Production {

   // A Production owns a local copy of an array of items,
   // where each item is a negative nonterminal index, or
   // positive terminal index. Attempts to index past the
   // end of the production yield a 0 item.

   friend class AllocReporter;

private:

   UINT m_nId;             // Unique integer id (0-based) of this production
   UINT m_nPriority;       // Relative priority of this production
   UINT m_n;               // Number of items in production
   INT* m_pList;           // List of items in production
   Production* m_pNext;    // Next production of same nonterminal

   static AllocCounter ac;

protected:

public:

   Production(UINT nId, UINT nPriority, UINT n, const INT* pList);

   ~Production(void);

   void setNext(Production* p);
   Production* getNext(void) const
      { return this->m_pNext; }

   UINT getId(void) const
      { return this->m_nId; }
   UINT getLength(void) const
      { return this->m_n; }
   BOOL isEpsilon(void) const
      { return this->m_n == 0; }
   UINT getPriority(void) const
      { return this->m_nPriority; }

   // Get the item at the dot position within the production
   INT operator[] (UINT nDot) const;

};


class Grammar {

   // A Grammar is a collection of Nonterminals
   // with their Productions.

   friend class AllocReporter;

private:

   UINT m_nNonterminals;   // Number of nonterminals
   UINT m_nTerminals;      // Number of terminals (indexed from 1)
   INT m_iRoot;            // Index of root nonterminal (negative)
   Nonterminal** m_nts;    // Array of Nonterminal pointers, owned by the Grammar class

   static AllocCounter ac;

protected:

public:

   Grammar(UINT nNonterminals, UINT nTerminals, INT iRoot = -1);
   Grammar(void);
   ~Grammar(void);

   void reset(void);

   BOOL readBinary(const CHAR* pszFilename);

   UINT getNumNonterminals(void) const
      { return this->m_nNonterminals; }
   UINT getNumTerminals(void) const
      { return this->m_nTerminals; }
   INT getRoot(void) const
      { return this->m_iRoot; }

   void setNonterminal(INT iIndex, Nonterminal*);

   Nonterminal* operator[] (INT iIndex) const;

   const WCHAR* nameOfNt(INT iNt) const;

};


class Label {

   // A Label is associated with a Node.

   friend class Node;

private:

   INT m_iNt;
   UINT m_nDot;
   Production* m_pProd;
   UINT m_nI;
   UINT m_nJ;
   UINT m_nToken;
   INT m_nTerminalScore;

public:

   Label(INT iNt, UINT nDot, Production* pProd, UINT nI, UINT nJ)
      : m_iNt(iNt), m_nDot(nDot), m_pProd(pProd), m_nI(nI), m_nJ(nJ)
      { }

   BOOL operator==(const Label& other) const
      { return ::memcmp((void*)this, (void*)&other, sizeof(Label)) == 0; }

   INT getSymbol();
   UINT getI();
   UINT getJ();
   Production* getProduction();
   UINT getDot();
   void setToken(UINT nToken);
   UINT getToken();
   void setTerminalScore(INT value);
   INT getTerminalScore();
};

// Callback function to Python to add a terminal to a set for a specific column / Earley set.
typedef BOOL (*AddTerminalToSetFunc)(UINT nHandle, UINT nColumnNumber, UINT nTerminalValue);

// Callback function to start scoring terminals associated with a column / Earley set
typedef BOOL (*StartScoringTerminalsForColumnFunc)(UINT nHandle, UINT nColumnNumber);

// Callback function to get the score for a terminal at a specific token position / column number.
typedef INT (*GetScoreForTerminalFunc)(UINT nHandle, UINT nTerminalValue, UINT nColumnNumber);

class Node {

   friend class AllocReporter;

private:

   struct FamilyEntry {
      Production* pProd;
      Node* p1;
      Node* p2;
      FamilyEntry* pNext;
      INT* pScore;
   };

   Label m_label;
   FamilyEntry* m_pHead;
   UINT m_nRefCount;
   BOOL m_bHasScore = false;
   UINT m_nHandle;
   Parser* m_pParser; // Reference to the parser

   static AllocCounter ac;

   void _dump(Grammar*, UINT nIndent);

protected:

public:

   Node(const Label&, Parser*, UINT nHandle);
   ~Node(void);

   void addRef(void)
      { this->m_nRefCount++; }
   void delRef(void);

   void addFamily(Production*, Node* pW, Node* pV, UINT i, INT nSymbolV, INT nSymbolW, State* pState, AddTerminalToSetFunc addTerminalToSetFunc, INT nHandle, Parser* parser);

   BOOL hasLabel(const Label& label) const
      { return this->m_label == label; }
   
   Label getLabel()
   {
      return this->m_label;
   }

   void dump(Grammar*);

   static UINT numCombinations(Node*);

   INT* getScore(UINT maxPosition);
   void doScore(UINT maxPosition);
};

class NodeDict2 {

public:

   NodeDict2(void);
   ~NodeDict2(void);

   struct NdEntry2 {
      Node* pNode;
      NdEntry2* pNext;
   };

   Node* next();
   
   void lookupOrAdd(Node* pNode);

   BOOL findAndDelete(Node* pNode);

   void reset(void);

   UINT getLength();

   Node* NodeDict2::getTopNodeAndDeleteFromDict();

private:
   
   NdEntry2* m_pHead;
   NdEntry2* m_pCurrent;
   UINT m_length;

};

class TauOrPsiDict{

public:

   TauOrPsiDict(void);
   ~TauOrPsiDict(void);

   struct TauOrPsiDictEntry {
      State* pState;
      TauOrPsiDictEntry* pNext;
   };

   State* next();
   TauOrPsiDictEntry* getHead();

   BOOL lookupOrAdd(State* pState);
   BOOL findAndDelete(State* pState);
   void resetCurrentToHead();
   void reset(void);

   UINT getLength();

private:

   TauOrPsiDictEntry* m_pHead;
   TauOrPsiDictEntry* m_pCurrent;
   UINT m_length;
};

// Token-terminal matching function
typedef BOOL (*MatchingFunc)(UINT nHandle, UINT nToken, UINT nTerminal);

// Allocator for token/terminal matching cache
typedef BYTE* (*AllocFunc)(UINT nHandle, UINT nToken, UINT nTerminals);

// Default matching function that simply
// compares the token value with the terminal number
BOOL defaultMatcher(UINT nHandle, UINT nToken, UINT nTerminal);


class Parser {

   // Earley-Scott parser for a given Grammar

   friend class AllocReporter;
   friend class Column;
   friend class Node;

private:

   // Grammar pointer, not owned by the Parser
   Grammar* m_pGrammar;
   MatchingFunc m_pMatchingFunc;
   AllocFunc m_pAllocFunc;
   AddTerminalToSetFunc m_pAddTerminalToSetFunc;
   StartScoringTerminalsForColumnFunc m_pStartScoringTerminalsForColumnFunc;
   GetScoreForTerminalFunc m_pGetScoreForTerminalFunc;
   NodeDict2 m_topNodesToTraverse;
   NodeDict2 m_childNodesToDelete;

   void push(UINT nHandle, State*, Column*, State*&, StateChunk*, UINT* pQLengthCounter);

   Node* makeNode(State* pState, UINT nEnd, Node* pV, NodeDict& ndV, UINT i, UINT nHandle);
   void helperAddLevel1PsiToPsiDict(UINT nOldCountE, UINT nOldCountQ, UINT* pQLengthCounter, Column* pEi, State* psNew);
   BOOL helperStateIsInPsiSet(State* pState, TauOrPsiDict* pPsiSet);

   // Internal token/terminal matching cache management
   BYTE* allocCache(UINT nHandle, UINT nToken, BOOL* pbNeedsRelease);
   void releaseCache(BYTE* abCache);

protected:

public:

   Parser(Grammar*, AddTerminalToSetFunc, StartScoringTerminalsForColumnFunc, GetScoreForTerminalFunc, MatchingFunc = defaultMatcher, AllocFunc = NULL);
   ~Parser(void);

   UINT getNumTerminals(void) const
      { return this->m_pGrammar->getNumTerminals(); }
   UINT getNumNonterminals(void) const
      { return this->m_pGrammar->getNumNonterminals(); }
   MatchingFunc getMatchingFunc(void) const
      { return this->m_pMatchingFunc; }
   GetScoreForTerminalFunc getGetScoreForTerminalFunc(void) const
      { return this->m_pGetScoreForTerminalFunc; }
   Grammar* getGrammar(void) const
      { return this->m_pGrammar; }

   // If pnToklist is NULL, a sequence of integers 0..nTokens-1 will be used
   Node* parse(UINT nHandle, INT iStartNt, UINT* pnErrorToken,
      UINT nTokens, const UINT pnToklist[] = NULL);

};

class Helper
{

public:

   static void printProduction(State* pState);
   static void printProduction(Production* pProd, INT lhs, INT nDot);
};

// Print a report on memory allocation
extern "C" void printAllocationReport(void);

// Parse a token stream
extern "C" Node* earleyParse(Parser*, UINT nTokens, INT iRoot, UINT nHandle, UINT* pnErrorToken);

extern "C" Grammar* newGrammar(const CHAR* pszGrammarFile);

extern "C" void deleteGrammar(Grammar*);

extern "C" Parser* newParser(Grammar*, AddTerminalToSetFunc fpAddTerminalToSetFunc, StartScoringTerminalsForColumnFunc fpStartScoringTerminalsForColumn, GetScoreForTerminalFunc fpGetScoreForTerminal, MatchingFunc fpMatcher = defaultMatcher, AllocFunc fpAlloc = NULL);

extern "C" void deleteParser(Parser*);

extern "C" void deleteForest(Node*);

extern "C" void dumpForest(Node*, Grammar*);

extern "C" UINT numCombinations(Node*);

