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
   It is designed to be called from Python code via CFFI with
   already parsed and packed grammar structures.

   The Earley parser used here is the improved version described by Scott & Johnstone,
   referencing Tomita. This allows worst-case cubic (O(n^3)) order, where n is the
   length of the input sentence, while still returning all possible parse trees
   for an ambiguous grammar.

   See Elizabeth Scott, Adrian Johnstone:
   "Recognition is not parsing — SPPF-style parsing from cubic recognisers"
   Science of Computer Programming, Volume 75, Issues 1–2, 1 January 2010, Pages 55–70

*/

// #define DEBUG

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <vector>

#include "eparser.h"


// Local implementation classes


class AllocReporter {

   // A debugging aid to diagnose and report memory leaks

private:
protected:
public:

   AllocReporter(void);
   ~AllocReporter(void);

   void report(void) const;

};

void printAllocationReport(void)
{
   AllocReporter reporter;
   reporter.report();
}


class State {

   // Parser state, contained within a Column

friend class AllocReporter;

private:

   INT m_iNt;              // Nonterminal (negative index)
   Production* m_pProd;    // Production
   UINT m_nDot;            // Dot (position in production)
   UINT m_nStart;          // Start token index
   Node* m_pw;             // Tree node
   State* m_pNext;         // Next state within column
   State* m_pNtNext;       // Next state with same Nt at prod[dot]

   static AllocCounter ac;

protected:

public:

   State(INT iNt, UINT nDot, Production* pProd, UINT nStart, Node* pw);
   State(State*, Node* pw);
   ~State(void);

   void increment(Node* pwNew);

   void setNext(State*);
   void setNtNext(State*);

   State* getNext(void) const
      { return this->m_pNext; }
   State* getNtNext(void) const
      { return this->m_pNtNext; }

   UINT getHash(void) const
      {
         return ((UINT)this->m_iNt) ^
            ((UINT)((uintptr_t)this->m_pProd) & 0xFFFFFFFF) ^
            (this->m_nDot << 7) ^ (this->m_nStart << 9) ^
            (((UINT)((uintptr_t)this->m_pw) & 0xFFFFFFFF) << 1);
      }
   BOOL operator==(const State& other) const
      {
         const State& t = *this;
         return t.m_iNt == other.m_iNt &&
            t.m_pProd == other.m_pProd &&
            t.m_nDot == other.m_nDot &&
            t.m_nStart == other.m_nStart &&
            t.m_pw == other.m_pw;
      }

   // Get the terminal or nonterminal at the dot
   INT prodDot(void) const
      { return (*this->m_pProd)[this->m_nDot]; }

   INT getNt(void) const
      { return this->m_iNt; }
   UINT getStart(void) const
      { return this->m_nStart; }
   UINT getDot(void) const
      { return this->m_nDot; }
   Production* getProd(void) const
      { return this->m_pProd; }
   Node* getNode(void) const
      { return this->m_pw; }
   Node* getResult(INT iStartNt) const;

};


class Column {

   // An Earley column
   // A Parser contains one Column for each token in the input, plus a sentinel

friend class AllocReporter;

private:

   // The contained States are stored an array of hash bins

   static const UINT HASH_BINS = 997; /* 499; */ // Prime number

   struct HashBin {
      State* m_pHead; // The first state in this hash bin
      State* m_pTail; // The last state in this hash bin
      State* m_pEnum; // The last enumerated state in this hash bin
   };

   Parser* m_pParser; // The associated parser
   UINT m_nToken; // The input token associated with this column
   State** m_pNtStates; // States linked by the nonterminal at their prod[dot]
   MatchingFunc m_pMatchingFunc; // Pointer to the token/terminal matching function
   BYTE* m_abCache; // Matching cache, a true/false flag for every terminal in the grammar
   BOOL m_bNeedsRelease; // Does the matching cache need to be explicitly released?
   HashBin m_aHash[HASH_BINS]; // The hash bin array
   UINT m_nEnumBin; // Round robin used during enumeration of states
   
   BenOrPsiDict m_oBenDict; // Dictionary of states / Earley items conforming to type BΣN (for scoring)
   UINT m_nLength; // The number of items in the Earley set / column
   BOOL m_bAreTerminalsScored; // Indicates if the terminals for this column have been scored yet
   BenOrPsiDict* m_pNonPsiDict; // Dictionary of states used as a HELPER or TEMPORARY LIST to find items to conforming to type ψ (small psi)
   BenOrPsiDict** m_pPsiDicts; // Dictionary of states / Earley items conforming to type ψ (small psi) that point, perhaps indirectly, 
                                      // to BΣN items within previous and current Earley sets / column which have propagated to the current Earley set / column.

   BOOL m_bDelayTerminalScoring;

   static AllocCounter ac;
   static AllocCounter acMatches;

protected:

public:

   Column(Parser*, UINT nToken);
   ~Column(void);

   UINT getToken(void) const
      { return this->m_nToken; }

   void startParse(UINT nHandle);
   void stopParse(void);

   // Add a state to the column, at the end of the state list
   BOOL addState(State* p);

   void resetEnum(void);
   State* nextState(void);

   State* getNtHead(INT iNt) const;

   BOOL matches(UINT nHandle, UINT nTerminal) const;

   BOOL areTerminalsScored();
   void setTerminalsScored();
   UINT getLength();
   BOOL correspondingBenStateExists(INT nt, UINT h);
   void addToNonPsiDict(State* pState);
   void addToPsiDicts(State* pState, UINT colNum);
   BenOrPsiDict* getNonPsiDictPointer();
   BenOrPsiDict* getPsiDicts(UINT colNum);
   BOOL shouldTerminalScoringBeDelayed();
   void setTerminalScoringShouldBeDelayed(BOOL val);
   void initializePsiSets(UINT cols);
};

class HNode {

   // Represents an element in the H set,
   // corresponding to a completed nullable
   // production of the associated nonterminal

friend class AllocReporter;

private:

   INT m_iNt;
   Node* m_pv;
   HNode* m_pNext;

   static AllocCounter ac;

protected:

public:

   HNode(INT iNt, Node* pv)
      : m_iNt(iNt), m_pv(pv)
      { HNode::ac++; }

   ~HNode()
      { HNode::ac--; }

   INT getNt(void) const
      { return this->m_iNt; }
   Node* getV(void) const
      { return this->m_pv; }
   HNode* getNext(void) const
      { return this->m_pNext; }
   void setNext(HNode* ph)
      { this->m_pNext = ph; }

};

AllocCounter HNode::ac;


class NodeDict {

   // Dictionary to map labels to node pointers

friend class AllocReporter;

private:

   struct NdEntry {
      Node* pNode;
      NdEntry* pNext;
   };

   NdEntry* m_pHead;

   static AllocCounter acLookups;

protected:

public:

   NodeDict(void);
   ~NodeDict(void);

   Node* lookupOrAdd(const Label&, Parser* pParser, UINT nHandle);

   void reset(void);

};

AllocCounter NodeDict::acLookups;


AllocCounter Nonterminal::ac;

Nonterminal::Nonterminal(const WCHAR* pwzName)
   : m_pwzName(NULL), m_pProd(NULL)
{
   Nonterminal::ac++;
   this->m_pwzName = pwzName ? ::wcsdup(pwzName) : NULL;
}

Nonterminal::~Nonterminal(void)
{
   if (this->m_pwzName)
      free(this->m_pwzName);
   // Delete the associated productions
   Production* p = this->m_pProd;
   while (p) {
      Production* pNext = p->getNext();
      delete p;
      p = pNext;
   }
   Nonterminal::ac--;
}

void Nonterminal::addProduction(Production* p)
{
   // Add a production at the head of the linked list
   p->setNext(this->m_pProd);
   this->m_pProd = p;
}


AllocCounter Production::ac;

Production::Production(UINT nId, UINT nPriority, UINT n, const INT* pList)
   : m_nId(nId), m_nPriority(nPriority), m_n(n), m_pList(NULL), m_pNext(NULL)
{
   Production::ac++;
   if (n > 0) {
      this->m_pList = new INT[n];
      ::memcpy((void*)this->m_pList, (void*)pList, n * sizeof(INT));
   }
}

Production::~Production(void) {
   // Destructor
   if (this->m_pList)
      delete [] this->m_pList;
   Production::ac--;
}

void Production::setNext(Production* p)
{
   this->m_pNext = p;
}

INT Production::operator[] (UINT nDot) const
{
   // Return the terminal or nonterminal at prod[dot]
   // or 0 if indexing past the end of the production
   return nDot < this->m_n ? this->m_pList[nDot] : 0;
}


// States are allocated in chunks, rather than individually
static const UINT CHUNK_SIZE = 2048 * sizeof(State);

struct StateChunk {

   StateChunk* m_pNext;
   UINT m_nIndex;
   BYTE m_ast[CHUNK_SIZE];

   StateChunk(StateChunk* pNext)
      : m_pNext(pNext), m_nIndex(0)
      { memset(this->m_ast, 0, CHUNK_SIZE); }

};

static AllocCounter acChunks;

void* operator new(size_t nBytes, StateChunk*& pChunkHead)
{
   ASSERT(nBytes == sizeof(State));
   // Allocate a new place for a state in a state chunk
   StateChunk* p = pChunkHead;
   if (!p || (p->m_nIndex + nBytes >= CHUNK_SIZE)) {
      StateChunk* pNew = new StateChunk(p);
      acChunks++;
      pChunkHead = p = pNew;
   }
   void* pPlace = (void*)(p->m_ast + p->m_nIndex);
   p->m_nIndex += (UINT)nBytes;
   ASSERT(p->m_nIndex <= CHUNK_SIZE);
   return pPlace;
}

static void freeStates(StateChunk*& pChunkHead)
{
   StateChunk* pChunk = pChunkHead;
   while (pChunk) {
      StateChunk* pNext = pChunk->m_pNext;
      delete pChunk;
      acChunks--;
      pChunk = pNext;
   }
   pChunkHead = NULL;
}

// Counter of states that are allocated and then immediately discarded
static UINT nDiscardedStates = 0;

AllocCounter State::ac;

State::State(INT iNt, UINT nDot, Production* pProd, UINT nStart, Node* pw)
   : m_iNt(iNt), m_pProd(pProd), m_nDot(nDot), m_nStart(nStart), m_pw(pw),
      m_pNext(NULL), m_pNtNext(NULL)
{
   State::ac++;
   if (pw)
      pw->addRef();
}

State::State(State* ps, Node* pw)
   : m_iNt(ps->m_iNt), m_pProd(ps->m_pProd), m_nDot(ps->m_nDot + 1),
      m_nStart(ps->m_nStart), m_pw(pw), m_pNext(NULL), m_pNtNext(NULL)
{
   // Create a new state by advancing one item forward from an existing state
   State::ac++;
   if (pw)
      pw->addRef();
}

State::~State(void)
{
   if (this->m_pw) {
      this->m_pw->delRef();
      this->m_pw = NULL;
   }
   State::ac--;
}

void State::increment(Node* pwNew)
{
   // 'Increment' the state, i.e. move the dot right by one step
   // and put in a new node pointer
   this->m_nDot++;
   this->m_pNext = NULL;
   ASSERT(this->m_pNtNext == NULL);
   if (pwNew)
      pwNew->addRef(); // Do this first, for safety
   if (this->m_pw)
      this->m_pw->delRef();
   this->m_pw = pwNew;
}

void State::setNext(State* p)
{
   this->m_pNext = p;
}

void State::setNtNext(State* p)
{
   this->m_pNtNext = p;
}

Node* State::getResult(INT iStartNt) const
{
   if (this->m_iNt == iStartNt && this->prodDot() == 0 &&
      this->m_nStart == 0)
      return this->m_pw;
   return NULL;
}


AllocCounter Column::ac;
AllocCounter Column::acMatches;

Column::Column(Parser* pParser, UINT nToken)
   : m_pParser(pParser),
      m_nToken(nToken),
      m_pNtStates(NULL),
      m_pMatchingFunc(pParser->getMatchingFunc()),
      m_abCache(NULL), m_bNeedsRelease(false),
      m_nEnumBin(0),
      m_bAreTerminalsScored(false)
{
   Column::ac++;
   ASSERT(this->m_pMatchingFunc != NULL);
   UINT nNonterminals = pParser->getNumNonterminals();
   // Initialize array of linked lists by nonterminal at prod[dot]
   this->m_pNtStates = new State* [nNonterminals];
   memset(this->m_pNtStates, 0, nNonterminals * sizeof(State*));
   // Initialize the hash bins to zero
   memset(this->m_aHash, 0, sizeof(HashBin) * HASH_BINS);

   this->m_oBenDict = BenOrPsiDict();
   this->m_pNonPsiDict = new BenOrPsiDict();
}

Column::~Column(void)
{
   // Destroy the states still owned by the column
   for (UINT i = 0; i < HASH_BINS; i++) {
      // Clean up each hash bin in turn
      HashBin* ph = &this->m_aHash[i];
      State* q = ph->m_pHead;
      while (q) {
         State* pNext = q->getNext();
         ASSERT(pNext != NULL || q == ph->m_pTail);
         // The states are allocated via placement new, so
         // they are not deleted ordinarily - we just run their destructor
         q->~State();
         q = pNext;
      }
      ph->m_pHead = NULL;
      ph->m_pTail = NULL;
   }
   // Delete array of linked lists by nonterminal at prod[dot]
   delete [] this->m_pNtStates;

   // Destroy the PsiDicts
   delete [] this->m_pPsiDicts;

   // Delete matching cache and seen array, if still allocated
   this->stopParse();
   Column::ac--;
}

void Column::startParse(UINT nHandle)
{
   // Called when the parser starts processing this column
   ASSERT(this->m_abCache == NULL);
   // Ask the parser to create a matching cache for us
   // (or eventually re-use a previous one)
   if (this->m_nToken != (UINT)-1)
      this->m_abCache = this->m_pParser->allocCache(nHandle, this->m_nToken, &this->m_bNeedsRelease);
}

void Column::stopParse(void)
{
   // Called when the parser is finished processing this column
   if (this->m_abCache && this->m_bNeedsRelease)
      // The matching cache needs to be released
      this->m_pParser->releaseCache(this->m_abCache);
   this->m_abCache = NULL;
}

BOOL Column::addState(State* p)
{
   // Check to see whether an identical state is
   // already present in the hash bin
   UINT nBin = p->getHash() % HASH_BINS;
   HashBin* ph = &this->m_aHash[nBin];
   State* q = ph->m_pHead;
   while (q) {
      if ((*q) == (*p))
         // Identical state: we're done
         return false;
      q = q->getNext();
   }
   // Not already found: link into place within the hash bin
   p->setNext(NULL);
   if (!ph->m_pHead) {
      // Establish linked list with one item
      ph->m_pHead = ph->m_pTail = p;
   }
   else {
      // Link the new element at the tail
      ph->m_pTail->setNext(p);
      ph->m_pTail = p;
   }
   // Get the item at prod[dot]
   INT iItem = p->prodDot();
   if (iItem < 0) {
      // Nonterminal: add to linked list
      UINT nIndex = ~((UINT)iItem);
      State*& psHead = this->m_pNtStates[nIndex];
      p->setNtNext(psHead);
      psHead = p;
   }
   // If we are adding state on the form BΣN, add it to the BenDict.
   // Note! Since Greynir departs from Scott's original paper by creating a token node just before the SCANNER which is 0 based, rather than 1 based, we must 
   // add 1 to the symbol for this to match correctly
   if(p->getNode() != NULL && p->getNode()->getLabel().getSymbol() + 1 == this->getToken() && p->prodDot() < 0)
   {
      this->m_oBenDict.lookupOrAdd(p);
      /*if(this->m_oBenDict.lookupOrAdd(p))
      {
         printf("BEN item added for token position %u. Node's symbol is %d, prodDot is: %d, PR is:\n", this->m_nToken, p->getNode()->getLabel().getSymbol() + 1, p->prodDot());
         Helper::printProduction(p);
      }*/
   }
   this->m_nLength++;
   return true;
}

State* Column::nextState(void)
{
   // Start our enumeration attempt from the last bin we looked at
   UINT n = this->m_nEnumBin;
   do {
      HashBin* ph = &this->m_aHash[n];
      if (!ph->m_pEnum && ph->m_pHead) {
         // Haven't enumerated from this one before,
         // but it has an entry: return it
         ph->m_pEnum = ph->m_pHead;
         this->m_nEnumBin = n;
         return ph->m_pEnum;
      }
      // Try the next item after the one we last returned
      State* pNext = ph->m_pEnum ? ph->m_pEnum->getNext() : NULL;
      if (pNext) {
         // There is such an item: return it
         ph->m_pEnum = pNext;
         this->m_nEnumBin = n;
         return pNext;
      }
      // Can't enumerate any more from this bin: go to the next one
      n = (n + 1) % HASH_BINS;
   } while (n != this->m_nEnumBin);
   // Gone full circle: Nothing more to enumerate
   return NULL;
}

void Column::resetEnum(void)
{
   // Start a fresh enumeration
   for (UINT i = 0; i < HASH_BINS; i++)
      this->m_aHash[i].m_pEnum = NULL;
   this->m_nEnumBin = 0;
}

State* Column::getNtHead(INT iNt) const
{
   UINT nIndex = ~((UINT)iNt);
   return this->m_pNtStates[nIndex];
}

BOOL Column::matches(UINT nHandle, UINT nTerminal) const
{
   if (this->m_nToken == (UINT)-1)
      // Sentinel token in last column never matches
      return false;
   ASSERT(this->m_abCache != NULL);
   if (this->m_abCache[nTerminal] & 0x80)
      // We already have a cached result for this terminal
      return (BOOL)(this->m_abCache[nTerminal] & 0x01);
   // Not cached: obtain a result and store it in the cache
   BOOL b = this->m_pMatchingFunc(nHandle, this->m_nToken, nTerminal) != 0;
   Column::acMatches++; // Count calls to the matching function
   // Mark our cache
   this->m_abCache[nTerminal] = b ? (BYTE)0x81 : (BYTE)0x80;
   return b;
}

BOOL Column::areTerminalsScored()
{
   return this->m_bAreTerminalsScored;
}

void Column::setTerminalsScored()
{
   this->m_bAreTerminalsScored = true;
}

UINT Column::getLength()
{
   return this->m_nLength;
}

BOOL Column::correspondingBenStateExists(INT nt, UINT h)
{
   if(this->m_oBenDict.getHead() == NULL) return false;

   this->m_oBenDict.resetCurrentToHead();

   while(State* pState = this->m_oBenDict.next())
   {
      if(nt == pState->prodDot() && h == this->getToken()) return true;
   }
   return false;
}

void Column::addToNonPsiDict(State* pState)
{
   this->m_pNonPsiDict->lookupOrAdd(pState);
}

BOOL Column::shouldTerminalScoringBeDelayed()
{
   return this->m_bDelayTerminalScoring;
}

void Column::setTerminalScoringShouldBeDelayed(BOOL val)
{
   this->m_bDelayTerminalScoring = val;
}

void Column::initializePsiSets(UINT cols)
{
   if(cols == 0) cols = 1;
   this->m_pPsiDicts = new BenOrPsiDict* [cols + 1];
   for(UINT j = 0; j < cols + 1; j++)
   {
      this->m_pPsiDicts[j] = new BenOrPsiDict();
   }
}

BenOrPsiDict* Column::getNonPsiDictPointer()
{
   return this->m_pNonPsiDict;
}

void Column::addToPsiDicts(State* pState, UINT nColNum)
{
   // We do not use PsiDict for column 0 but initialize it for simplicities' sake
   // (otherwise we'll have to deal with lots on null reference errors)
   this->m_pPsiDicts[nColNum]->lookupOrAdd(pState);
}

BenOrPsiDict* Column::getPsiDicts(UINT nColNum)
{
   // We do not use PsiDict for column 0 but initialize it for simplicities' sake
   // (otherwise we'll have to deal with lots on null reference errors)
   return this->m_pPsiDicts[nColNum];
}

class File {

   // Safe wrapper for FILE*

private:

   FILE* m_f;

public:

   File(const CHAR* pszFilename, const CHAR* pszMode)
      { this->m_f = fopen(pszFilename, pszMode); }
   ~File(void)
      { if (this->m_f) fclose(this->m_f); }

   operator FILE*() const
      { return this->m_f; }
   operator BOOL() const
      { return this->m_f != NULL; }

   UINT read(void* pb, UINT nLen)
      { return this->m_f ? (UINT)fread(pb, 1, nLen, this->m_f) : 0; }
   UINT write(void* pb, UINT nLen)
      { return this->m_f ? (UINT)fwrite(pb, 1, nLen, this->m_f) : 0; }

   BOOL read_UINT(UINT& n)
      { return this->read(&n, sizeof(UINT)) == sizeof(UINT); }
   BOOL read_INT(INT& i)
      { return this->read(&i, sizeof(INT)) == sizeof(INT); }

};


AllocCounter Grammar::ac;

Grammar::Grammar(UINT nNonterminals, UINT nTerminals, INT iRoot)
   : m_nNonterminals(nNonterminals), m_nTerminals(nTerminals), m_iRoot(iRoot), m_nts(NULL)
{
   Grammar::ac++;
   this->m_nts = new Nonterminal*[nNonterminals];
   memset(this->m_nts, 0, nNonterminals * sizeof(Nonterminal*));
}

Grammar::Grammar(void)
   : m_nNonterminals(0), m_nTerminals(0), m_iRoot(0), m_nts(NULL)
{
   Grammar::ac++;
}

Grammar::~Grammar(void)
{
   this->reset();
   Grammar::ac--;
}

void Grammar::reset(void)
{
   for (UINT i = 0; i < this->m_nNonterminals; i++)
      if (this->m_nts[i])
         delete this->m_nts[i];
   if (this->m_nts) {
      delete [] this->m_nts;
      this->m_nts = NULL;
   }
   this->m_nNonterminals = 0;
   this->m_nTerminals = 0;
   this->m_iRoot = 0;
}

class GrammarResetter {

   // Resets a grammar to a known zero state unless
   // explicitly disarmed

private:

   Grammar* m_pGrammar;

public:

   GrammarResetter(Grammar* pGrammar)
      : m_pGrammar(pGrammar)
      { }
   ~GrammarResetter(void)
      { if (this->m_pGrammar) this->m_pGrammar->reset(); }

   void disarm(void)
      { this->m_pGrammar = NULL; }

};

BOOL Grammar::readBinary(const CHAR* pszFilename)
{
   // Attempt to read grammar from binary file.
   // Returns true if successful, otherwise false.
#ifdef DEBUG   
   printf("Reading binary grammar file %s\n", pszFilename);
#endif
   this->reset();
   File f(pszFilename, "rb");
   if (!f)
      return false;
   const UINT SIGNATURE_LENGTH = 16;
   BYTE abSignature[SIGNATURE_LENGTH];
   UINT n = f.read(abSignature, sizeof(abSignature));
   if (n < sizeof(abSignature))
      return false;
   // Check the signature - should start with 'Greynir'
   if (memcmp(abSignature, "Greynir", 7) != 0) {
#ifdef DEBUG
      printf("Signature mismatch\n");
#endif      
      return false;
   }
   UINT nNonterminals, nTerminals;
   if (!f.read_UINT(nTerminals))
      return false;
   if (!f.read_UINT(nNonterminals))
      return false;
#ifdef DEBUG   
   printf("Reading %u terminals and %u nonterminals\n", nTerminals, nNonterminals);
#endif
   if (!nNonterminals)
      // No nonterminals to read: we're done
      return true;
   INT iRoot;
   if (!f.read_INT(iRoot))
      return false;
#ifdef DEBUG   
   printf("Root nonterminal index is %d\n", iRoot);
#endif
   // Initialize the nonterminals array
   Nonterminal** ppnts = new Nonterminal*[nNonterminals];
   memset(ppnts, 0, nNonterminals * sizeof(Nonterminal*));
   this->m_nts = ppnts;
   this->m_nNonterminals = nNonterminals;
   this->m_nTerminals = nTerminals;
   this->m_iRoot = iRoot;
   // Ensure we clean up properly in case of exit with error
   GrammarResetter resetter(this);
   // Loop through the nonterminals
   for (n = 0; n < nNonterminals; n++) {
      // How many productions?
      UINT nLenPlist;
      if (!f.read_UINT(nLenPlist))
         return false;
      Nonterminal* pnt = new Nonterminal(L"");
      // Loop through the productions
      for (UINT j = 0; j < nLenPlist; j++) {
         UINT nId;
         if (!f.read_UINT(nId))
            return false;
         UINT nPriority;
         if (!f.read_UINT(nPriority))
            return false;
         UINT nLenProd;
         if (!f.read_UINT(nLenProd))
            return false;
         const UINT MAX_LEN_PROD = 256;
         if (nLenProd > MAX_LEN_PROD) {
            // Production too long
#ifdef DEBUG            
            printf("Production too long\n");
#endif            
            return false;
         }
         // Read the production
         INT aiProd[MAX_LEN_PROD];
         f.read(aiProd, nLenProd * sizeof(INT));
         // Create a fresh production object
         Production* pprod = new Production(nId, nPriority, nLenProd, aiProd);
         // Add it to the nonterminal
         pnt->addProduction(pprod);
      }
      // Add the nonterminal to the grammar
      this->setNonterminal(-1 -(INT)n, pnt);
   }
#ifdef DEBUG   
   printf("Reading completed\n");
   fflush(stdout);
#endif
   // No error: we disarm the resetter
   resetter.disarm();
   return true;
}

void Grammar::setNonterminal(INT iIndex, Nonterminal* pnt)
{
   // iIndex is negative
   ASSERT(iIndex < 0);
   UINT nIndex = ~((UINT)iIndex); // -1 becomes 0, -2 becomes 1, etc.
   ASSERT(nIndex < this->m_nNonterminals);
   if (nIndex < this->m_nNonterminals)
      this->m_nts[nIndex] = pnt;
}

Nonterminal* Grammar::operator[] (INT iIndex) const
{
   // Return the nonterminal with index nIndex (1-based)
   ASSERT(iIndex < 0);
   UINT nIndex = ~((UINT)iIndex); // -1 becomes 0, -2 becomes 1, etc.
   return (nIndex < this->m_nNonterminals) ? this->m_nts[nIndex] : NULL;
}

const WCHAR* Grammar::nameOfNt(INT iNt) const
{
   Nonterminal* pnt = (*this)[iNt];
   return pnt ? pnt->getName() : L"[None]";
}


AllocCounter Node::ac;

INT Label::getSymbol()
{
   return this->m_iNt;
}

UINT Label::getI()
{
   return this->m_nI;
}

UINT Label::getJ()
{
   return this->m_nJ;
}

Production* Label::getProduction()
{
   return this->m_pProd;
}

UINT Label::getDot()
{
   return this->m_nDot;
}

void Label::setToken(UINT nToken)
{
   this->m_nToken = nToken;
}

UINT Label::getToken()
{
   return this->m_nToken;
}

void Label::setTerminalScore(INT value)
{
   this->m_nTerminalScore = value;
}

INT Label::getTerminalScore()
{
   return this->m_nTerminalScore;
}

// Node constructor now needs a refrence to the parser and the handle for the terminal scoring feature
// to be able to call the necessary functions on the Python side.
Node::Node(const Label& label, Parser* parser, UINT nHandle)
   : m_label(label), m_pHead(NULL), m_nRefCount(1), m_pParser(parser), m_nHandle(nHandle)
{
   Node::ac++;
}

Node::~Node(void)
{
   FamilyEntry* p = this->m_pHead;
   while (p) {
      FamilyEntry* pNext = p->pNext;
      if (p->p1)
         p->p1->delRef();
      if (p->p2)
         p->p2->delRef();
      delete p;
      p = pNext;
   }
   this->m_pHead = NULL;
   Node::ac--;
}

void Node::delRef(void)
{
   ASSERT(this->m_nRefCount > 0);
   if (!--this->m_nRefCount)
      delete this;
}

// TODO: Remove pState later as it is only used for debugging
void Node::addFamily(Production* pProd, Node* pW, Node* pV, UINT i, INT nSymbolV, INT nSymbolW, State* pState, AddTerminalToSetFunc fpAddTerminalToSetFunc, INT nHandle, Parser* parser)
{
   // pW may be NULL, or both may be NULL if epsilon
   FamilyEntry* p = this->m_pHead;
   while (p) {
      if (p->pProd == pProd && p->p1 == pW && p->p2 == pV)
         // We already have the same family entry
         return;
      p = p->pNext;
   }
   // Not already there: create a new entry
   p = new FamilyEntry();
   p->pProd = pProd;

   // RB: Adding feature to swap out token nodes for terminal nodes.
   bool wasChild = false; // TODO: Remove later, variable is just used for debugging

   if(pW)
   {
      Label tokenLabelW = pW->getLabel();
      INT nChildSymbolW = tokenLabelW.getSymbol(); // nChildSymbol will only be different from nProdSymbol in the case of a token/terminal

      if(nChildSymbolW >= 0) // Note that the token symbol can be 0 whereas terminal symbol would be not (unless it is an epsilon terminal)
      {
         wasChild = true;
         bool success = fpAddTerminalToSetFunc(nHandle, tokenLabelW.getI(), nSymbolW);
         if(!success) printf("fpAddTerminalToSetFunc returned False for node pW. This should not happen.\n");
         Label labelW(nSymbolW, 0, NULL, tokenLabelW.getI(), tokenLabelW.getJ());
         labelW.setToken(nChildSymbolW); // Token information is needed on the Python side
         Node* terminalNodeW = new Node(labelW, parser, nHandle);
         p->p1 = terminalNodeW; // No need to add reference here as it happens automatically when the new terminal node is created here above.
         // printf("addFamily pW - Added terminal %d to terminals set for column/getI: %u. getJ: %u, i: %u\n", nSymbolW, tokenLabelW.getI(), tokenLabelW.getJ(), i);
         // Helper::printProduction(pState);
      }
      else
      {
         p->p1 = pW;
         pW->addRef();
      }
   }
   else
   {
      p->p1 = pW;
   }

   if(pV)
   {
      Label tokenLabelV = pV->getLabel();
      INT nChildSymbolV = tokenLabelV.getSymbol(); // nChildSymbol will only be different from nProdSymbol in the case of a token/terminal

      if(nChildSymbolV >= 0)
      {
         wasChild = true;
         bool success = fpAddTerminalToSetFunc(nHandle, tokenLabelV.getI(), nSymbolV);
         if(!success) printf("fpAddTerminalToSetFunc returned False for node pV. This should not happen.\n");
         Label labelV(nSymbolV, 0, NULL, tokenLabelV.getI(), tokenLabelV.getJ());
         labelV.setToken(nChildSymbolV); // Token information is need on the Python side
         Node* terminalNodeV = new Node(labelV, parser, nHandle);
         p->p2 = terminalNodeV; // No need to add reference here as it happens automatically when the new terminal node is created here above.
         // printf("addFamily pV - Added terminal %d to terminals set for column/getI: %u, getJ: %u, i: %u.\n", nSymbolV, tokenLabelV.getI(), tokenLabelV.getJ(), i);
         // Helper::printProduction(pState);
      }
      else
      {
         p->p2 = pV;
         pV->addRef();
      }
   }
   else
   {
      p->p2 = pV;
   }
   
   // if(wasChild) Helper::printProduction(pState); // TODO: Just used for debugging - remove later, including pState parameter.
   p->pNext = this->m_pHead;
   this->m_pHead = p;
}

void Node::_dump(Grammar* pGrammar, UINT nIndent)
{
   for (UINT i = 0; i < nIndent; i++)
      printf("  ");
   Production* pProd = this->m_label.m_pProd;
   UINT nDot = this->m_label.m_nDot;
   INT iDotProd = pProd ? (*pProd)[nDot] : 0;
   INT iNt = this->m_label.m_iNt;
   const WCHAR* pwzName;
   WCHAR wchBuf[16];
   if (iNt < 0) {
      // Nonterminal
      pwzName = pGrammar->nameOfNt(iNt);
      if (!pwzName || wcslen(pwzName) == 0) {
         swprintf(wchBuf, 16, L"(Nt %d)", iNt);
         pwzName = wchBuf;
      }
   }
   else {
      // Token
      swprintf(wchBuf, 16, L"(Token %d)", iNt);
      pwzName = wchBuf;
   }
   printf("Label: %ls %u %d [%u:%u]\n",
      pwzName,
      nDot,
      iDotProd,
      this->m_label.m_nI,
      this->m_label.m_nJ);
   FamilyEntry* p = this->m_pHead;
   UINT nOption = 0;
   while (p) {
      if (nOption || p->pNext) {
         // Don't print 'Option 1' if there is only one option
         for (UINT i = 0; i < nIndent; i++)
            printf("  ");
         printf("Option %u\n", nOption + 1);
      }
      if (p->p1)
         p->p1->_dump(pGrammar, nIndent + 1);
      if (p->p2)
         p->p2->_dump(pGrammar, nIndent + 1);
      p = p->pNext;
      nOption++;
   }
   fflush(stdout);
}

void Node::dump(Grammar* pGrammar)
{
   this->_dump(pGrammar, 0);
}

UINT Node::numCombinations(Node* pNode)
{
   // Be careful when calling this function: it may take a long
   // time to execute since it visits nodes for every possible
   // combination of nodes in the tree. In other words, it ignores
   // the packing of the tree. The Python version keeps track of
   // already visited nodes and is much more efficient for large
   // trees.
   if (!pNode || pNode->m_label.m_iNt >= 0)
      return 1;
   UINT nComb = 0;
   FamilyEntry* p = pNode->m_pHead;
   while (p) {
      UINT n1 = p->p1 ? Node::numCombinations(p->p1) : 1;
      UINT n2 = p->p2 ? Node::numCombinations(p->p2) : 1;
      nComb += n1 * n2;
      p = p->pNext;
   }
   return nComb == 0 ? 1 : nComb;
}

// Gets a pointer to the score if it exists, NULL otherwise.
INT* Node::getScore(UINT maxPosition)
{
   // If this is a terminal
   // As the terminal columns on the Python side are 0 based we only get < maxPosition. Otherwise we might ask for terminals that do not exist yet due to BΣN items
   if(this->m_label.getSymbol() > 0 && this->m_label.getI() < maxPosition) 
   {
      // printf("Trying to call getGetScoreForTerminalFunc in the parser this time. Is null: %s\n", this->m_pParser->getGetScoreForTerminalFunc() == NULL ? "true" : "false");
      // printf("m_nHandle: %d, symbol: %d, i: %d.\n", this->m_nHandle, this->m_label.getSymbol(), this->m_label.getI());
      INT temp = (this->m_pParser->getGetScoreForTerminalFunc())(this->m_nHandle, this->m_label.getSymbol(), this->m_label.getI());
      // printf("Creating a pointer to value %d and returning the pointer.\n", temp);
      INT* retVal = new INT(temp);
      this->m_label.setTerminalScore(*retVal);
      return retVal;
   }
   // if this is a non-terminal and it has a score already
   else if(this->m_label.getSymbol() < 0 && this->m_bHasScore && this->m_label.getI() <= maxPosition)
   {
      // printf("We have a score of %d, returning it!\n", *this->m_pHead->pScore);
      return this->m_pHead->pScore; // return the score of remaining family / packed node
   }
   else
   {
      // printf("Returning NULL\n");
      return NULL; // otherwise we do not have a score yet so we return NULL.
   }
}

void Node::doScore(UINT maxPosition, UINT level)
{
   if(this->m_label.getSymbol() < 0 && !this->m_bHasScore && this->m_label.m_nI <= maxPosition)
   {
      FamilyEntry* p = this->m_pHead;
      INT familyCounter = 1;
      INT* p1Score = NULL;
      INT* p2Score = NULL;
      while(p)
      {
         if(p->p1)
         {
            p->p1->doScore(maxPosition, level + 1);
            p1Score = p->p1->getScore(maxPosition);
         }
         if(p->p2)
         {
            p->p2->doScore(maxPosition, level + 1);
            p2Score = p->p2->getScore(maxPosition);
         }
         
         if(p->p1 && p->p2 && p1Score && p2Score)
         {
            // we can add the scores for each and give the family a score which is their sum.
            p->pScore = new INT(*p1Score + *p2Score);                                          
         }
         else if(p->p1 == NULL && p->p2 && p2Score)
         {
            p->pScore = p2Score;
         }
         else if(p->p1 == NULL && p->p2 == NULL)
         {
            // This is a wildcard non-terminal (* or ?) which has no children
            // We give the family a score of 0
            p->pScore = new INT(0);
         }
         
         p = p->pNext;

         familyCounter++;
      }
      // Loop again through all families. If they have all been scored we can set
      // this->m_bHasScore to true and drop lower scoring families.
      p = this->m_pHead;
      BOOL anyUnscored = false;
      FamilyEntry* highestScoringFE = NULL;
      INT highestScore = -10000; // Just something low enough as scores can be negative but not that negative.
      while(p)
      {
         if(p->pScore == NULL) 
         {  
            anyUnscored = true;
            break;
         }
         else
         {
            // TODO: Think about when there is a tie and how do resolve that non-deterministically
            // as per suggestion from Miðeind.
            if(*p->pScore > highestScore)
            {
               highestScore = *p->pScore;
               highestScoringFE = p;
            }
         }
         p = p->pNext;
      }
      if(!anyUnscored) 
      {
         this->m_bHasScore = true;

         // Throw away lower scoring families
         p = this->m_pHead;
         while (p) {
            FamilyEntry* pNext = p->pNext;
            if(highestScoringFE == p)
            {  
               p->pNext = NULL;
               this->m_pHead = p;
            }
            else
            {
               // remember to include <string> if you run this test again.
               if (p->p1)
                  p->p1->delRef(); // This is ok
               if (p->p2)
                  p->p2->delRef(); // This is ok
               delete p;
            }
            p = pNext;
         }
      }  
   }
}

NodeDict::NodeDict(void)
   : m_pHead(NULL)
{
}

NodeDict::~NodeDict(void)
{
   this->reset();
}

Node* NodeDict::lookupOrAdd(const Label& label, Parser* parser, UINT nHandle)
{
   // If the label is already found in the NodeDict,
   // return the corresponding node.
   // Otherwise, create a new node, add it to the dict
   // under the label, and return it.
   NodeDict::acLookups++;
   NdEntry* p = this->m_pHead;
   while (p) {
      if (p->pNode->hasLabel(label))
         return p->pNode;
      p = p->pNext;
   }
   // Not found: add to the dict
   p = new NdEntry();
   p->pNode = new Node(label, parser, nHandle);
   p->pNext = this->m_pHead;
   this->m_pHead = p;
   return p->pNode;
}

void NodeDict::reset(void)
{
   NdEntry* p = this->m_pHead;
   while (p) {
      NdEntry* pNext = p->pNext;
      p->pNode->delRef();
      delete p;
      p = pNext;
   }
   this->m_pHead = NULL;
}

// NodeDict2 is only used for the scoring feature. It should not add or delete references to the nodes.

NodeDict2::NodeDict2(void)
   : m_pHead(NULL), m_pCurrent(NULL), m_length(0)
{
}

NodeDict2::~NodeDict2(void)
{
   this->reset();
}

void NodeDict2::lookupOrAdd(Node* pNode)
{
   NdEntry2* p = this->m_pHead;
   while (p) {
      if (p->pNode->hasLabel((*pNode).getLabel()))
         return;
      p = p->pNext;
   }
   // Not found: add to the dict
   p = new NdEntry2();
   p->pNode = pNode;
   p->pNext = this->m_pHead;
   this->m_pHead = p;
   this->m_length++;
   return;
}

/*
   Returns the next node in the dictionary. Returns NULL if it is at the end and then starts
   at the beginning if called again.
*/
Node* NodeDict2::next()
{
   if(this->m_pCurrent == NULL) // We are at the beginning or NodeDict2 is empty
      this->m_pCurrent = this->m_pHead; // In case of empty then m_pHead will be NULL also
   else
   {
      this->m_pCurrent = this->m_pCurrent->pNext;
   }
   if(this->m_pCurrent == NULL) return NULL; // NodeDict2 is empty or at the end
   else return this->m_pCurrent->pNode;
}

// Just deletes the reference to the node in the dictionary, not the actual node.
BOOL NodeDict2::findAndDelete(Node* pNode)
{
   NdEntry2* p = this->m_pHead;
   if(p->pNode->hasLabel((*pNode).getLabel()))
   {
      this->m_pHead = p->pNext;
      delete p;
      this->m_length--;
      return true;
   }
   else
   {
      NdEntry2* lastEntry = p;
      p = p->pNext;
      while (p) {
         if(p->pNode->hasLabel((*pNode).getLabel()))
         {
            NdEntry2* pNext = p->pNext;
            delete p;
            lastEntry->pNext = pNext;
            this->m_length--;
            return true;
         }
         lastEntry = p;
         p = p->pNext;
      }

      return false; // Node was not found, return false;
   }
}

Node* NodeDict2::getTopNodeAndDeleteFromDict()
{
   if(this->m_length == 0) return NULL; // If the dictionary is empty

   NdEntry2* p = this->m_pHead;
   Node* topNode = p->pNode;
   p = p->pNext;
   while (p) {
      INT currentNodeLength = p->pNode->getLabel().getJ() - p->pNode->getLabel().getI();
      INT topNodeLength = topNode->getLabel().getJ() - topNode->getLabel().getI();
      if(currentNodeLength > topNodeLength) topNode = p->pNode;
      p = p->pNext;
   }
   this->findAndDelete(topNode);
   return topNode;
}

void NodeDict2::reset(void)
{
   NdEntry2* p = this->m_pHead;
   while (p) {
      NdEntry2* pNext = p->pNext;
      delete p;
      p = pNext;
   }
   this->m_pHead = NULL;
   this->m_length = 0;
}

UINT NodeDict2::getLength()
{
   return this->m_length;
}

BenOrPsiDict::BenOrPsiDict(void)
   : m_pHead(NULL), m_pCurrent(NULL), m_length(0)
{
}

BenOrPsiDict::~BenOrPsiDict(void)
{
   this->reset();
}

BenOrPsiDict::BenOrPsiDictEntry* BenOrPsiDict::getHead()
{
   return this->m_pHead;
}

BOOL BenOrPsiDict::lookupOrAdd(State* pState)
{
   BenOrPsiDictEntry* p = this->m_pHead;
   while (p)
   {
      if(p->pState == pState) return false;
      p = p->pNext;
   }
   // Not found: add to the dict
   p = new BenOrPsiDictEntry();
   p->pState = pState;
   p->pNext = this->m_pHead;
   this->m_pHead = p;
   this->m_length++;
   return true;  
}

/* 
   Just deletes the reference to the Earley item / state in the dictionary, not the actual node.
*/
BOOL BenOrPsiDict::findAndDelete(State* pState)
{
   BenOrPsiDictEntry* p = this->m_pHead;
   if(p->pState == pState)
   {
      this->m_pHead = p->pNext;
      delete p;
      this->m_length--;
      return true;
   }
   else
   {
      BenOrPsiDictEntry* lastEntry = p;
      p = p->pNext;
      while (p) {
         if(p->pState == pState)
         {
            BenOrPsiDictEntry* pNext = p->pNext;
            delete p;
            lastEntry->pNext = pNext;
            this->m_length--;
            return true;
         }
         lastEntry = p;
         p = p->pNext;
      }

      return false; // State was not found, return false;
   }
}

State* BenOrPsiDict::next()
{
   if(this->m_pCurrent == NULL) // We are at the beginning or BenOrPsiDict is empty
   {
      this->m_pCurrent = this->m_pHead; // In case of empty then m_pHead will be NULL also
   }
   else
   {
      this->m_pCurrent = this->m_pCurrent->pNext;
   }
   if(this->m_pCurrent == NULL) return NULL;
   else return this->m_pCurrent->pState;
}

/*
   Delete all entries and start with a clean slate
*/ 
void BenOrPsiDict::reset(void)
{
   BenOrPsiDictEntry* p = this->m_pHead;
   while(p) {
      BenOrPsiDictEntry* pNext = p->pNext;
      delete p;
      p = pNext;
   }
   this->m_pHead = NULL;
   this->m_length = 0;
}

UINT BenOrPsiDict::getLength()
{
   return this->m_length;
}

// Causes next() to start with the head in case m_pCurrent was somewhere in the middle.
void BenOrPsiDict::resetCurrentToHead()
{
   this->m_pCurrent = NULL;
}

Parser::Parser(Grammar* p, AddTerminalToSetFunc fpAddTerminalToSetFunc, StartScoringTerminalsForColumnFunc fpStartScoringTerminalsForColumn, GetScoreForTerminalFunc fpGetScoreForTerminalFunc, MatchingFunc pMatchingFunc, AllocFunc pAllocFunc)
   : m_pGrammar(p), m_pAddTerminalToSetFunc(fpAddTerminalToSetFunc), m_pStartScoringTerminalsForColumnFunc(fpStartScoringTerminalsForColumn), m_pGetScoreForTerminalFunc(fpGetScoreForTerminalFunc),  m_pMatchingFunc(pMatchingFunc), m_pAllocFunc(pAllocFunc)
{
   ASSERT(this->m_pGrammar != NULL);
   ASSERT(this->m_pMatchingFunc != NULL);
   ASSERT(this->m_pAddTerminalToSetFunc != NULL);
   ASSERT(this->m_pStartScoringTerminalsForColumnFunc != NULL);
   ASSERT(this->m_pGetScoreForTerminalFunc != NULL);
}

Parser::~Parser(void)
{
}

BYTE* Parser::allocCache(UINT nHandle, UINT nToken, BOOL* pbNeedRelease)
{
   // Create a fresh token/terminal matching cache
   UINT nTerminals = this->getNumTerminals();
   BYTE* abCache = NULL;
   *pbNeedRelease = false;
   if (this->m_pAllocFunc)
      // There is a cache allocation function: call it
      abCache = this->m_pAllocFunc(nHandle, nToken, nTerminals + 1);
   if (!abCache) {
      // Either no cache allocation function, or it returned NULL
      // Allocate our own buffer and initialize it to zero
      abCache = new BYTE[nTerminals + 1];
      memset(abCache, 0, (nTerminals + 1) * sizeof(BYTE));
      *pbNeedRelease = true;
   }
   return abCache;
}

void Parser::releaseCache(BYTE* abCache)
{
   // Release a token/terminal matching cache
   // (Only call if *pbNeedRelease was true after calling allocCache())
   delete [] abCache;
}

Node* Parser::makeNode(State* pState, UINT nEnd, Node* pV, NodeDict& ndV, UINT i, UINT nHandle)
{
   UINT nDot = pState->getDot() + 1;
   Production* pProd = pState->getProd();
   UINT nLen = pProd->getLength();
   if (nDot == 1 && nLen >= 2)
      return pV;

   INT iNtB = pState->getNt();
   UINT nStart = pState->getStart();
   Node* pW = pState->getNode();
   Production* pProdLabel = pProd;

   INT nSymbolV = NULL;
   if(pV)
   {
      nSymbolV = (*pProd)[nDot - 1];
      /*if(nSymbolV > 0)
      {
         printf("makeNode - created nSymbolV = %d and position: %u, i: %u\n", nSymbolV, nDot - 1, i);
         Helper::printProduction(pState);
      }*/
   }

   INT nSymbolW = NULL;
   if(pW)
   {
      nSymbolW = (*pProd)[nDot - 2];
      /*if(nSymbolW > 0)
      {
         printf("makeNode - created nSymbolW = %d, position: %u, i: %u\n", nSymbolW, nDot - 2, i);
         Helper::printProduction(pState);
      }*/
   }

   if (nDot >= nLen) { // RB: if β = eps { let s = B } í ritgerð
      // Completed production: label by nonterminal only
      nDot = 0;
      pProdLabel = NULL; 
   }
   Label label(iNtB, nDot, pProdLabel, nStart, nEnd);
   Node* pY = ndV.lookupOrAdd(label, this, nHandle);
   
   // Add the parent node pY to the set / dictionary of top nodes which will start traversing into later
   this->m_topNodesToTraverse.lookupOrAdd(pY);

   // Add the child nodes pV and/or pW to the set / dictionary of child Nodes which we will use to delete from the 
   // set of top nodes later. When traversing the nodes we will only be interested in traversing top nodes and can therefore 
   // discard all child nodes.
   if(pW) this->m_childNodesToDelete.lookupOrAdd(pW);
   if(pV) this->m_childNodesToDelete.lookupOrAdd(pV);

   pY->addFamily(pProd, pW, pV, i, nSymbolV, nSymbolW, pState, this->m_pAddTerminalToSetFunc, nHandle, this); // pW may be NULL
   return pY;
}

void Parser::push(UINT nHandle, State* pState, Column* pE, State*& pQ, StateChunk* pChunkHead, UINT* pQLengthCounter)
{
   INT iItem = pState->prodDot();
   if (iItem <= 0) {
      // Nonterminal or epsilon: add state to column
      if (pE->addState(pState))
         // State did not already exist in the column: we're done
         return;
   }
   else
   if (pE->matches(nHandle, (UINT)iItem)) {
      // Terminal matching the current token
      // Link into list whose head is pQ
      pState->setNext(pQ);
      pQ = pState;
      if(pQLengthCounter != NULL) *pQLengthCounter++;
      return;
   }
   // We did not actually push the state; discard it
   pState->~State();
   ASSERT(pChunkHead != NULL);
   ASSERT(pChunkHead->m_nIndex >= sizeof(State));
   if ((BYTE*)pState + sizeof(State) == pChunkHead->m_ast + pChunkHead->m_nIndex) {
      // The state is the most recently allocated one in the chunk
      // (a very common case): go back one location in the chunk
      pChunkHead->m_nIndex -= sizeof(State);
      nDiscardedStates++;
   }
}

Node* Parser::parse(UINT nHandle, INT iStartNt, UINT* pnErrorToken,
   UINT nTokens, const UINT pnToklist[])
{
   // If pnToklist is NULL, a sequence of integers 0..nTokens-1 will be used
   // Sanity checks
   if (!nTokens)
      return NULL;
   if (!this->m_pGrammar)
      return NULL;
   if (iStartNt >= 0)
      // Root must be nonterminal (index < 0)
      return NULL;
   Nonterminal* pRootNt = (*this->m_pGrammar)[iStartNt];
   if (!pRootNt)
      // No or invalid root nonterminal
      return NULL;
   if (pnErrorToken)
      *pnErrorToken = 0;

   // Initialize the Earley columns
   UINT i;
   Column** pCol = new Column* [nTokens + 1];
   for (i = 0; i < nTokens; i++)
   {
      pCol[i] = new Column(this, pnToklist ? pnToklist[i] : i);
      pCol[i]->initializePsiSets(i);
   }
      
   pCol[i] = new Column(this, (UINT)-1); // Sentinel column
   pCol[i]->initializePsiSets(i);

   // Initialize parser state
   State* pQ0 = NULL;
   StateChunk* pChunkHead = NULL;

   // Prepare the the first column
   pCol[0]->startParse(nHandle);

   // Prepare the initial state
   Production* p = pRootNt->getHead();
   while (p) {
      State* ps = new (pChunkHead) State(iStartNt, 0, p, 0, NULL);
#ifdef DEBUG
      printf("For initial state, pushing production starting with nonterminal %d\n", (INT)(*p)[0]);
#endif
      this->push(nHandle, ps, pCol[0], pQ0, pChunkHead, NULL);
      p = p->getNext();
   }

   // Main parse loop
   State* pQ = NULL;
   NodeDict ndV; // Node dictionary
   UINT nNumNonterminals = this->getNumNonterminals();
   BYTE* pbSeen = new BYTE[nNumNonterminals];

#ifdef DEBUG
   clock_t clockStart = clock();
   clock_t clockLast = clockStart;
#endif

   // TERMINAL SCORING ADDITION STARTS
   // Counters declared
   UINT* pQLengthCounter;
   UINT nOldCountQ;
   UINT nOldCountE;
   // TERMINAL SCORING ADDITION ENDS

   for (i = 0; i < nTokens + 1; i++) {

      Column* pEi = pCol[i];
      pEi->resetEnum();
      State* pState = pEi->nextState();

      // TERMINAL SCORING ADDITION STARTS
      // Counters reset in each iteration
      pQLengthCounter = new UINT(0);
      UINT nOldCountQ = 0;
      UINT nOldCountE = 0;
      // printf("STARTING ROUND %u. Token is %u\n", i, pEi->getToken());
      // TERMINAL SCORING ADDITION ENDS

#ifdef DEBUG
      printf("Column %u, token %d\n", i, pEi->getToken());
#endif

      if (!pState && !pQ0) {
         // No parse available at token i-1
         if (pnErrorToken)
            *pnErrorToken = i;
         break;
      }

      pQ = pQ0;
      pQ0 = NULL;
      HNode* pH = NULL;
      
      // TERMINAL SCORING CHANGE STARTS
      // Count items in pQ
      while(pQ)
      {
         (*pQLengthCounter)++;
         pQ = pQ->getNext();
      }
      // TERMINAL SCORING CHANGE ENDS
      
      // No nonterminals seen yet
      memset(pbSeen, 0, nNumNonterminals * sizeof(BYTE));

      while (pState) {

         INT iItem = pState->prodDot();

         if (iItem < 0) {
            // Nonterminal at the dot: Earley predictor
            // Don't push the same nonterminal more than once to the same column
            if (!pbSeen[~((UINT)iItem)]) {
               // Earley predictor
               // Push all right hand sides of this nonterminal
               pbSeen[~((UINT)iItem)] = 1;
               p = (*this->m_pGrammar)[iItem]->getHead();
               while (p) {
                  State* psNew = new (pChunkHead) State(iItem, 0, p, i, NULL);

                  // TERMINAL SCORING CHANGE STARTS
                  nOldCountE = pEi->getLength();
                  nOldCountQ = *pQLengthCounter;
                  this->push(nHandle, psNew, pEi, pQ, pChunkHead, pQLengthCounter);
                  if(i > 0 && i < nTokens) this->helperAddLevel1PsiToPsiDict(nOldCountE, nOldCountQ, pQLengthCounter, pEi, psNew);
                  // TERMINAL SCORING CHANGE ENDS

                  p = p->getNext();
               }
            }
            // Add elements from the H set that refer to the
            // nonterminal iItem (nt_C)
            // NOTE: this code should NOT be within the above if(!pbSeen[...])
            HNode* ph = pH;
            while (ph) {
               if (ph->getNt() == iItem) {
                  Node* pY = this->makeNode(pState, i, ph->getV(), ndV, i, nHandle);
                  State* psNew = new (pChunkHead) State(pState, pY);

                  // TERMINAL SCORING CHANGE STARTS
                  nOldCountE = pEi->getLength();
                  nOldCountQ = *pQLengthCounter;
                  this->push(nHandle, psNew, pEi, pQ, pChunkHead, pQLengthCounter);
                  if(i > 0 && i < nTokens) this->helperAddLevel1PsiToPsiDict(nOldCountE, nOldCountQ, pQLengthCounter, pEi, psNew);
                  // TERMINAL SCORING CHANGE ENDS
               }
               ph = ph->getNext();
            }
         }
         else
         if (iItem == 0) {
            // Production completed: Earley completer
            INT iNtB = pState->getNt();
            UINT nStart = pState->getStart();
            Node* pW = pState->getNode();
            if (!pW) {
               Label label(iNtB, 0, NULL, i, i);
               pW = ndV.lookupOrAdd(label, this, nHandle);
               pW->addFamily(pState->getProd(), NULL, NULL, i, NULL, NULL, pState, this->m_pAddTerminalToSetFunc, nHandle, this); // Epsilon production
            }
            if (nStart == i) {
               HNode* ph = new HNode(iNtB, pW);
               ph->setNext(pH);
               pH = ph;
            }
            State* psNt = pCol[nStart]->getNtHead(iNtB);
            UINT completerCounter = 0;
            while (psNt) {
               Node* pY = this->makeNode(psNt, i, pW, ndV, i, nHandle);
               State* psNew = new (pChunkHead) State(psNt, pY);

               // TERMINAL SCORING CHANGE STARTS
               nOldCountE = pEi->getLength();
               nOldCountQ = *pQLengthCounter;
               this->push(nHandle, psNew, pEi, pQ, pChunkHead, NULL);
               // Check for level 1  psi items in the Psi set for the current Earley set / column
               if(i > 0 && i < nTokens) this->helperAddLevel1PsiToPsiDict(nOldCountE, nOldCountQ, pQLengthCounter, pEi, psNew); 

               // Check for psi items that have propagated to this Earley set and put them in the correct Psi set.
               // We only need to check for this if h (nStart) is greater than 0 as Earley set 0 does not have a Psi set.
               if(nStart > 0 && nStart < i && (pEi->getLength() > nOldCountE || *pQLengthCounter > nOldCountQ)) 
               {
                  for(UINT j = 1; j < nStart + 1; j++) // Check all Psi sets in column E_h
                  {
                     BenOrPsiDict* psiDict = pCol[nStart]->getPsiDicts(j);
                     psiDict->resetCurrentToHead();
                     
                     while(State* pSt3 = psiDict->next())
                     {
                        if(pSt3->getNt() == psNew->getNt() && pSt3->getProd() == psNew->getProd())
                        {
                           completerCounter++;
                           pEi->addToPsiDicts(psNew, j);
                        }
                     }
                  }
               }
               // TERMINAL SCORING CHANGE ENDS

               psNt = psNt->getNtNext();
            }
         }

         // Move to the next item on the agenda
         // (which may have been enlarged by the previous code)
         pState = pEi->nextState();

      }
      // printf("After while consuming items from queue R - The following Psi sets are in column %u containing the following number of items:\n", i);
      /*printf("PSISETS: ");
      for(UINT j=0; j < i + 1; j++)
      {
         printf("[%u]: %u, ",j, pEi->getPsiDicts(j)->getLength());
      }
      printf(".\n");*/

      // Clean up the H set
      while (pH) {
         HNode* ph = pH->getNext();
         delete pH;
         pH = ph;
      }

      // Reset the node dictionary
      ndV.reset();
      Node* pV = NULL;

      // Done processing this column: let it clean up
      pEi->stopParse();

      if (pQ) {
         Label label(pEi->getToken(), 0, NULL, i, i + 1);
         pV = new Node(label, this, nHandle); // Reference is deleted below
         // Open up the next column
         pCol[i + 1]->startParse(nHandle);
      }

      // TERMINAL SCORING CHANGE STARTS
      if(i < nTokens) // No need to do this in the last iteration since all token nodes in BΣN items will be adopted now if the token sequence is in the language of the CFG
      {
         // printf("BEFORE SCANNER: Before starting to look for upper level psi items. Will populate NonPsiDict now.\n");
         BOOL bUpperLevelPsiStateFound;
         for(UINT j = 1; j < i + 1; j++)
         {
            // Reset NonPsiDict and repopulate it only with new items from this Earley set / column 
            pEi->getNonPsiDictPointer()->reset();
            pEi->resetEnum();
            while(State* pSt5 = pEi->nextState())
            {
               if(pSt5->getStart() == i)
               {
                  pEi->getNonPsiDictPointer()->lookupOrAdd(pSt5);
               }
            }
            State* pQcopy = pQ;
            while(pQcopy)
            {
               if(pQcopy->getStart() == i)
               {
                  pEi->getNonPsiDictPointer()->lookupOrAdd(pQcopy);
               }
               pQcopy = pQcopy->getNext();
            }

            // printf("Starting for loop. Iteration %u, j: %u.\n", i, j);
            UINT counter = 0;
            do{
               // Check new items against items already in this Earley sets' / columns' Psi sets. If any of them point to them then they are upper level psi items and added
               // to the correspoinding Psi set
               bUpperLevelPsiStateFound = false;
               pEi->getNonPsiDictPointer()->resetCurrentToHead();
               // printf("Going to loop through nonPsiDictsPointer which contains %u items.\n", pEi->getNonPsiDictPointer()->getLength());
               UINT itemCount = 0;
               while(State* pNonPsiState = pEi->getNonPsiDictPointer()->next())
               {
                  // Look at PsiDict in this Earley set (column)
                  //printf("Looking into column %u. State: \n", nColNum);
                  //Helper::printProduction(pNonPsiState);
                  BenOrPsiDict* pPsiDict_j_InCurrent = pEi->getPsiDicts(j);
                  pPsiDict_j_InCurrent->resetCurrentToHead();
                  //printf("Going to loop through states in PsiDict for column %u, of length %u.\n", j, pPsiDict_j_InCurrent->getLength());
                  while(State* pSt4 = pPsiDict_j_InCurrent->next())
                  {
                     if(pNonPsiState->getNt() == pSt4->prodDot())
                     {
                        // printf("Found state ");
                        // Helper::printProduction(pSt4);

                        // Add this upper level psi item to the Psi set for corresponding column
                        pPsiDict_j_InCurrent->lookupOrAdd(pNonPsiState);
                        
                        // Remove from the non-psiState dictionary / linked list
                        pEi->getNonPsiDictPointer()->findAndDelete(pNonPsiState);
                        bUpperLevelPsiStateFound = true;
                        itemCount++;
                     }
                  }
               }
               // printf("i: %u, j: %u, -- %u psi items in level %u found.\n", i, j, itemCount, ++counter + 1);
               
            } while(bUpperLevelPsiStateFound);
         }
         
         // printf("BEFORE SCANNER: Done looking for upper level psi items. NonPsiDict length: %u.\n", pEi->getNonPsiDictPointer()->getLength());
      }
      // TERMINAL SCORING CHANGE ENDS

      *pQLengthCounter = 0;
      nOldCountQ = 0;
      if(i < nTokens) nOldCountE = pCol[i + 1]->getLength();

      while (pQ) {
         // Earley scanner
         State* psNext = pQ->getNext();
         Node* pY = this->makeNode(pQ, i + 1, pV, ndV, i, nHandle);
         // Instead of throwing away the old state and creating
         // a new almost identical one, re-use the old after
         // 'incrementing' it by moving the dot one step to the right
         pQ->increment(pY);
         ASSERT(i + 1 <= nTokens);
         this->push(nHandle, pQ, pCol[i + 1], pQ0, pChunkHead, pQLengthCounter);      
         pQ = psNext;
      }

      /*printf("AFTER SCANNER - The following Psi sets are in column %u containing the following number of items:\n", i);
      printf("PSISETS: ");
      for(UINT j=0; j < i + 1; j++)
      {
         printf("[%u]: %u, ", j, pEi->getPsiDicts(j)->getLength());
      }
      printf(".\n");*/

      // if(i < nTokens) printf("Added %u items to E_%d and %u items to Q' after SCANNER.\n", pCol[i + 1]->getLength() - nOldCountE, i + 1, *pQLengthCounter);

      // TERMINAL SCORING CHANGE STARTS
      if(i > 0 && i < nTokens)
      {
         pCol[i + 1]->resetEnum();
         while(State* pSt2 = pCol[i + 1]->nextState())
         {
            // printf("Checking if the state added to E_%u", i + 1);
            // Helper::printProduction(pSt2);
            // printf("exists in Psi sets of current column %u.\n", i);
            for(UINT j = 1; j < i + 1; j++)
            {
               pCol[j]->setTerminalScoringShouldBeDelayed(false); // Set this to false everywhere initially.
               if(pCol[j]->areTerminalsScored() == false && helperStateIsInPsiSet(pSt2, pEi->getPsiDicts(j)))
               {
                  pCol[j]->setTerminalScoringShouldBeDelayed(true);
                  pCol[i + 1]->getPsiDicts(j)->lookupOrAdd(pSt2);
                  // printf("Yes, in Psi set for column %d.\n", j);
               }
               /*else
               {
                  printf("No, as far as items in set E_%d are concerned, not in E_%u's Psi set for column %d.\n", i + 1, i, j);
               }*/
            }
         }
         while(pQ0)
         {
            for(UINT j = 1; j < i + 1; j++)
            {
               if(pCol[j]->areTerminalsScored() == false && pCol[j]->shouldTerminalScoringBeDelayed() == false && helperStateIsInPsiSet(pQ0, pEi->getPsiDicts(j)))
               {
                  pCol[j]->setTerminalScoringShouldBeDelayed(true);
                  pCol[i + 1]->getPsiDicts(j)->lookupOrAdd(pQ0);
                  // printf("Yes, in Psi set for column %d.\n", j);
               }
               else
               {
                  // printf("No, as far as items in set Q' are concerned, not in E_%u's Psi set for column %d.\n", i, j);
               }
            }
            pQ0 = pQ0->getNext();
         }
         /*printf("AFTER DELAY LOGIC - The following Psi sets are in column %u containing the following number of items:\n", i+1);
         printf("PSISETS: ");
         for(UINT j=0; j < i + 2; j++)
         {
            printf("[%u]: %u, ",j, pCol[i + 1]->getPsiDicts(j)->getLength());
         }
         printf(".\n");*/
      }
      else if(i == nTokens)// We are in the last round, now all terminals can be scored
      {
         for(int j = i; j > 0; j--) // The zero column is omitted as it will not be terminal scored.
         {
            pCol[j]->setTerminalScoringShouldBeDelayed(false); // Set this to false everywhere now in the last iteration.
         }
      }
      /*else if(i == 0)
      {
         printf("Psi dict is empty in iteration 0.\n");
      }*/

      // Score the terminals
      UINT nMaxPositionToScore = 0;
      if(i > 0 && i < nTokens)
      {
         // if(i==1) printf("Not scoring yet for i = 1 since we are 1-based on the C++ side when scoring the terminals. We will earliest score when i==2.\n");
         for(UINT j = 1; j < i; j++)  // We only terminal-score the previous column provided the terminal scoring should not be further delayed due to BΣN and Psi items
         {
            // printf("ROUND %u. Attempting to Score column %u:\n", i, j);
            // printf("pCol[%u]->areTerminalsScored() = %s, pCol[%u]->shouldTerminalScoringBeDelayed() = %s\n", j, pCol[j]->areTerminalsScored() ? "true" : "false", j, pCol[j]->shouldTerminalScoringBeDelayed() ? "true": "false");
            if(pCol[j]->areTerminalsScored() == false && !pCol[j]->shouldTerminalScoringBeDelayed())
            {
               // printf("Scoring terminals ...\n");
               this->m_pStartScoringTerminalsForColumnFunc(nHandle, j-1); // Token position is 0 based on the Python side
               pCol[j]->setTerminalsScored();
            }
            else
            {
               // if(pCol[j]->areTerminalsScored() == true) printf("Column's terminals already scored.\n");
               // else printf("Scoring needs to be delayed. Propagated psi-items detected.");

               // Maximum position to score when scoring non-terminal nodes is the token position where we first encounter a column where
               // we must delay the scoring of terminals due to the existens of items on the form BΣN
               if(nMaxPositionToScore == 0) 
               {
                  nMaxPositionToScore = j-1; 
                  // printf( "Max position set to %u\n", nMaxPositionToScore);
               }
            }
         } 

         // This printout debugging only works with text2 as the columns are hard-coded
         // printf("Columns scored: [0]: %s, [1]: %s, [2]: %s, [3]: %s, [4]: %s, [5]: %s, [6]: %s, [7]: %s.\n", pCol[0]->areTerminalsScored() ? "true" : "false",
         // pCol[1]->areTerminalsScored() ? "true" : "false", pCol[2]->areTerminalsScored() ? "true" : "false", pCol[3]->areTerminalsScored() ? "true" : "false", pCol[4]->areTerminalsScored() ? "true" : "false",
         // pCol[5]->areTerminalsScored() ? "true" : "false", pCol[6]->areTerminalsScored() ? "true" : "false", pCol[7]->areTerminalsScored() ? "true" : "false");     
      }
      else if(i == nTokens)
      {
         for(int j = 1; j <= i; j++) 
         {
            // printf( "LAST ROUND %u. Now to score all remaining columns. Column %u:\n", i, j);
            // printf( "column %d. pCol[j]->areTerminalsScored() = %s, pCol[j]->shouldTerminalScoringBeDelayed() = %s\n", j, pCol[j]->areTerminalsScored() ? "true" : "false", pCol[j]->shouldTerminalScoringBeDelayed() ? "true": "false");
            if(pCol[j]->areTerminalsScored() == false)
            {
               // printf( "Scoring terminals ...\n");
               this->m_pStartScoringTerminalsForColumnFunc(nHandle, j-1); // Token position is 0 based on the Python side
               pCol[j]->setTerminalsScored();
            }
            else
            {
               // printf( "Criteria not met. Not scoring this time.\n");

               // Maximum position to score when scoring non-terminal nodes is the token position where we first encounter a column where
               // we must delay the scoring of terminals due to the existens of items on the form BΣN

               nMaxPositionToScore = i; 
               // printf( "Max position set to %u\n", nMaxPositionToScore);
            }
         } 

         // This printout debugging only works with text2 as the columns are hard-coded
         // printf("Columns scored: [0]: %s, [1]: %s, [2]: %s, [3]: %s, [4]: %s, [5]: %s, [6]: %s, [7]: %s.\n", pCol[0]->areTerminalsScored() ? "true" : "false",
         // pCol[1]->areTerminalsScored() ? "true" : "false", pCol[2]->areTerminalsScored() ? "true" : "false", pCol[3]->areTerminalsScored() ? "true" : "false", pCol[4]->areTerminalsScored() ? "true" : "false",
         // pCol[5]->areTerminalsScored() ? "true" : "false", pCol[6]->areTerminalsScored() ? "true" : "false", pCol[7]->areTerminalsScored() ? "true" : "false");          
      }
      // TERMINAL SCORING CHANGE ENDS

      // NON-TERMINAL SCORING STARTS
      if(i > 0)
      {
         // printf("Before delete. Length of m_topNodesToTraverse: %u, Length of m_childNodesToDelete: %u\n", this->m_topNodesToTraverse.getLength(),

         // Delete the child nodes before we start scoring non-terminal nodes as we will
         // be proceeding top down.
         while(Node* pNode = this->m_childNodesToDelete.next())
         {
            this->m_topNodesToTraverse.findAndDelete(pNode);
         }

         // printf("After delete. Length of m_topNodesToTraverse: %u, Length of m_childNodesToDelete: %u\n", this->m_topNodesToTraverse.getLength(),
         //   this->m_childNodesToDelete.getLength());   
         if(pCol[i-1]->areTerminalsScored() && nMaxPositionToScore > 0)
         {
            // printf( "Attempting to score nodes ...\n");
            // Find the top-most node
            while(Node* nodeToScore = this->m_topNodesToTraverse.getTopNodeAndDeleteFromDict())
            {
               nodeToScore->doScore(nMaxPositionToScore, 1);
            }
         }
         /*else
         {
            printf( "Not scoring nodes as terminals in column %u are not scored yet.\n", i-1);
         }*/
      }
      // NON-TERMINAL SCORING ENDS

      // Clean up reference to pV created above
      if (pV)
         pV->delRef();
      
      // printf("PARSER FINISHED ROUND %d\n", i);

   // if(i == nTokens) Helper::printSets(pCol, i);

#ifdef DEBUG
      clock_t clockNow = clock();
      clock_t clockElapsed = clockNow - clockStart;
      clock_t clockThis = clockNow - clockLast;
      clockLast = clockNow;
      printf ("Column %u finished in %.3f sec, elapsed %.3f sec\n", i,
         ((float)clockThis) / CLOCKS_PER_SEC, ((float)clockElapsed) / CLOCKS_PER_SEC);
      fflush(stdout);
#endif
   }

#ifdef DEBUG
   clock_t clockNow = clock() - clockStart;
   printf("Parse loop finished, elapsed %.3f sec\n",
      ((float)clockNow) / CLOCKS_PER_SEC);
#endif

   ASSERT(pQ == NULL);
   ASSERT(pQ0 == NULL);

   Node* pResult = NULL;
   if (i > nTokens) {
      // Completed the token loop
      pCol[nTokens]->resetEnum();
      State* ps = pCol[nTokens]->nextState();
      while (ps && !pResult) {
         // Look through the end states until we find one that spans the
         // entire parse tree and derives the starting nonterminal
         pResult = ps->getResult(iStartNt);
         if (pResult)
            // Save the result node from being deleted when the
            // column states are deleted
            pResult->addRef();
         else
            ps = pCol[nTokens]->nextState();
      }
      if (!pResult && pnErrorToken)
         // No parse available at the last column
         *pnErrorToken = nTokens;
   }

#ifdef DEBUG
   clockNow = clock() - clockStart;
   printf("Result found, elapsed %.3f sec\n",
      ((float)clockNow) / CLOCKS_PER_SEC);
#endif

   // Cleanup
   delete[] pbSeen;
   for (i = 0; i < nTokens + 1; i++)
      delete pCol[i];
   delete [] pCol;

   freeStates(pChunkHead);

#ifdef DEBUG
   clockNow = clock() - clockStart;
   printf("Cleanup finished, elapsed %.3f sec\n",
      ((float)clockNow) / CLOCKS_PER_SEC);
   //if (pResult)
   //   pResult->dump(this->m_pGrammar);
#endif

   return pResult; // The caller should call delRef() on this after using it
}

void Parser::helperAddLevel1PsiToPsiDict(UINT nOldCountE, UINT nOldCountQ, UINT* pQLengthCounter, Column* pEi, State* psNew)
{
   if(pEi->getLength() > nOldCountE || *pQLengthCounter > nOldCountQ)
   {
      // Earley item (state) was added. Check if it was on the form Psi level 1
      // i.e. it point directly to a BΣN item
      if(pEi->correspondingBenStateExists(psNew->getNt(), pEi->getToken()))
      {
         pEi->getPsiDicts(pEi->getToken())->lookupOrAdd(psNew);
      }
   }
}

BOOL Parser::helperStateIsInPsiSet(State* pState, BenOrPsiDict* pPsiSet)
{
   pPsiSet->resetCurrentToHead();
   while(State* pStateFromPsiSet = pPsiSet->next())
   {
      // TODO: Sanity check this helper. Skrýtið ef ekkert færist yfir. 
      if(pState->getNt() == pStateFromPsiSet->getNt() && pState->getProd() == pStateFromPsiSet->getProd())
         //&& pState->getStart() == pStateFromPsiSet->getStart())
      {
         return true;
      }
   }
   return false;
}


AllocReporter::AllocReporter(void)
{
}

AllocReporter::~AllocReporter(void)
{
}

void AllocReporter::report(void) const
{
   printf("\nMemory allocation status\n");
   printf("------------------------\n");
   printf("Nonterminals    : %6d %8d\n", Nonterminal::ac.getBalance(), Nonterminal::ac.numAllocs());
   printf("Productions     : %6d %8d\n", Production::ac.getBalance(), Production::ac.numAllocs());
   printf("Grammars        : %6d %8d\n", Grammar::ac.getBalance(), Grammar::ac.numAllocs());
   printf("Nodes           : %6d %8d\n", Node::ac.getBalance(), Node::ac.numAllocs());
   printf("States          : %6d %8d\n", State::ac.getBalance(), State::ac.numAllocs());
   printf("...discarded    : %6s %8d\n", "", nDiscardedStates);
   printf("StateChunks     : %6d %8d\n", acChunks.getBalance(), acChunks.numAllocs());
   printf("Columns         : %6d %8d\n", Column::ac.getBalance(), Column::ac.numAllocs());
   printf("HNodes          : %6d %8d\n", HNode::ac.getBalance(), HNode::ac.numAllocs());
   printf("NodeDict lookups: %6s %8d\n", "", NodeDict::acLookups.numAllocs());
   printf("Matching calls  : %6s %8d\n", "", Column::acMatches.numAllocs());
   fflush(stdout); // !!! Debugging
}


// The functions below are declared extern "C" for external invocation
// of the parser (e.g. from CFFI)

// Token-terminal matching function
BOOL defaultMatcher(UINT nHandle, UINT nToken, UINT nTerminal)
{
   // printf("defaultMatcher(): token is %u, terminal is %u\n", nToken, nTerminal);
   return nToken == nTerminal;
}

Grammar* newGrammar(const CHAR* pszGrammarFile)
{
   if (!pszGrammarFile)
      return NULL;
   // Read grammar from binary file
   Grammar* pGrammar = new Grammar();
   if (!pGrammar->readBinary(pszGrammarFile)) {
#ifdef DEBUG      
      printf("Unable to read binary grammar file %s\n", pszGrammarFile);
#endif      
      delete pGrammar;
      return NULL;
   }
   return pGrammar;
}

void deleteGrammar(Grammar* pGrammar)
{
   if (pGrammar)
      delete pGrammar;
}

Parser* newParser(Grammar* pGrammar, AddTerminalToSetFunc fpAddTerminalToSetFunc, StartScoringTerminalsForColumnFunc fpStartScoringTerminalsForColumn, GetScoreForTerminalFunc fpGetScoreForTerminal, MatchingFunc fpMatcher, AllocFunc fpAlloc)
{
   if (!pGrammar || !fpMatcher || !fpAddTerminalToSetFunc || !fpStartScoringTerminalsForColumn || !fpGetScoreForTerminal)
      return NULL;
   return new Parser(pGrammar, fpAddTerminalToSetFunc, fpStartScoringTerminalsForColumn, fpGetScoreForTerminal, fpMatcher, fpAlloc);
}

void deleteParser(Parser* pParser)
{
   if (pParser)
      delete pParser;
}

void deleteForest(Node* pNode)
{
   if (pNode)
      pNode->delRef();
}

void dumpForest(Node* pNode, Grammar* pGrammar)
{
   if (pNode && pGrammar)
      pNode->dump(pGrammar);
}

UINT numCombinations(Node* pNode)
{
   return pNode ? Node::numCombinations(pNode) : 0;
}

Node* earleyParse(Parser* pParser, UINT nTokens, INT iRoot, UINT nHandle, UINT* pnErrorToken)
{
   // Preparation and sanity checks
   if (!nTokens)
      return NULL;
   if (!pParser)
      return NULL;
   Grammar* pGrammar = pParser->getGrammar();
   if (!pGrammar)
      return NULL;
   if (iRoot == 0)
      // Root not specified: Run parser from the default root
      iRoot = pGrammar->getRoot();
   if (iRoot >= 0)
      return NULL; // Root must be a nonterminal (i.e. have a negative index)
   if (pnErrorToken)
      *pnErrorToken = 0;

#ifdef DEBUG
   printf("Calling pParser->parse()\n"); fflush(stdout);
#endif
   Node* pNode = pParser->parse(nHandle, iRoot, pnErrorToken, nTokens);
#ifdef DEBUG
   printf("Back from pParser->parse()\n"); fflush(stdout);
#endif

   return pNode;
}

void Helper::printProduction(State* pState)
{
   INT nNt = pState->getNt();
   Production* pProd = pState->getProd();
   INT prodLength = pProd->getLength();
   if(prodLength == 0) return; // We are not interested in productions for epsilon nodes as they are always empty
   INT nDot = pState->getDot();
   printf("(");
   Helper::printProduction(pProd, nNt, nDot);
   printf(", h=%u, ", pState->getStart());
   if(pState->getNode() != NULL)
   {
      printf("node=(%d, %u, %u), prodDot: %d)\n", pState->getNode()->getLabel().getSymbol(), pState->getNode()->getLabel().getI(), pState->getNode()->getLabel().getJ(), pState->prodDot());
   }
   else
   {
      printf("node=(null). prodDot: %d).\n", pState->prodDot());
   }
}

void Helper::printProduction(Production* pProd, INT lhs, INT nDot)
{
   std::vector<INT> vProductions;
   UINT prodLength = pProd->getLength();
   for(UINT i = 0; i < prodLength; i++)
   {
      vProductions.push_back((*pProd)[i]);
   }

   printf("PR: dot: %d, length: %d, prodVector:       %d -> ", nDot, prodLength, lhs);
   for(int i = 0; i < vProductions.size(); i++)
      printf("%d | ", vProductions[i]);
}