/* The following code was generated by JFlex 1.4.1 on 10. 1. 12 ���� 10:32 */

package org.apache.lucene.analysis.kr;

/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;


/**
 * This class is a scanner generated by 
 * <a href="http://www.jflex.de/">JFlex</a> 1.4.1
 * on 10. 1. 12 ���� 10:32 from the specification file
 * <tt>D:/eclipse-workspace/search/kr.analyzer/src/org/apache/lucene/analysis/kr/KoreanTokenizerImpl.jflex</tt>
 */
class KoreanTokenizerImpl {

  /** This character denotes the end of file */
  public static final int YYEOF = -1;

  /** initial size of the lookahead buffer */
  private static final int ZZ_BUFFERSIZE = 16384;

  /** lexical states */
  public static final int YYINITIAL = 0;

  /** 
   * Translates characters to character classes
   */
  private static final String ZZ_CMAP_PACKED = 
    "\11\0\1\0\1\17\1\0\1\0\1\16\22\0\1\0\5\0\1\3"+
    "\1\1\4\0\1\7\1\5\1\2\1\7\12\11\6\0\1\4\32\10"+
    "\4\0\1\6\1\0\32\10\105\0\27\10\1\0\37\10\1\0\u0568\10"+
    "\12\12\206\10\12\12\u026c\10\12\12\166\10\12\12\166\10\12\12\166\10"+
    "\12\12\166\10\12\12\167\10\11\12\166\10\12\12\166\10\12\12\166\10"+
    "\12\12\340\10\12\12\166\10\12\12\u0166\10\12\12\266\10\u0100\14\u0e00\10"+
    "\u1040\0\u0150\15\140\0\20\15\u0100\0\200\15\200\0\u19c0\15\100\0\u5200\15"+
    "\u0c00\0\u2bb0\13\u2150\0\u0200\15\u0465\0\73\15\75\10\43\0";

  /** 
   * Translates characters to character classes
   */
  private static final char [] ZZ_CMAP = zzUnpackCMap(ZZ_CMAP_PACKED);

  /** 
   * Translates DFA states to action switch labels.
   */
  private static final int [] ZZ_ACTION = zzUnpackAction();

  private static final String ZZ_ACTION_PACKED_0 =
    "\1\0\2\1\3\2\2\3\1\4\1\1\1\5\6\0"+
    "\3\2\1\3\1\5\3\0\1\2\1\3\1\5\1\2"+
    "\4\3\3\0\1\3\1\6\3\7\2\10\2\0\2\5"+
    "\2\0\2\7\1\3\4\5\1\0\2\7\1\3\4\0"+
    "\2\3\1\11\1\0\1\7\1\12\2\0\2\7\1\3"+
    "\1\11\1\0\2\5\1\3\2\5\1\3\2\12\1\7"+
    "\2\3\1\5\2\3\1\5\2\3\2\11\1\0\2\7"+
    "\1\0\1\12\1\7\2\5\1\12\2\0\1\12\1\3"+
    "\1\7\1\13\1\0\1\3\1\0\1\3\1\12\1\0"+
    "\1\3\1\0\1\3\1\0\1\7\1\3\1\5\1\3"+
    "\1\5\1\3\1\7\2\5\1\12\2\0\2\3\1\0"+
    "\2\3\1\0\2\3\1\12\1\7\1\3\2\0\2\3"+
    "\1\12\1\0\2\7\2\0\1\12\1\0\2\7\1\3"+
    "\2\0\1\3\1\7\1\0\1\12\1\0\1\12\1\3"+
    "\1\0\3\3";

  private static int [] zzUnpackAction() {
    int [] result = new int[172];
    int offset = 0;
    offset = zzUnpackAction(ZZ_ACTION_PACKED_0, offset, result);
    return result;
  }

  private static int zzUnpackAction(String packed, int offset, int [] result) {
    int i = 0;       /* index in packed string  */
    int j = offset;  /* index in unpacked array */
    int l = packed.length();
    while (i < l) {
      int count = packed.charAt(i++);
      int value = packed.charAt(i++);
      do result[j++] = value; while (--count > 0);
    }
    return j;
  }


  /** 
   * Translates a state to a row index in the transition table
   */
  private static final int [] ZZ_ROWMAP = zzUnpackRowMap();

  private static final String ZZ_ROWMAP_PACKED_0 =
    "\0\0\0\20\0\40\0\60\0\100\0\120\0\140\0\160"+
    "\0\200\0\220\0\240\0\260\0\300\0\320\0\340\0\360"+
    "\0\u0100\0\u0110\0\u0120\0\u0130\0\u0140\0\u0150\0\u0160\0\u0170"+
    "\0\u0180\0\u0190\0\u01a0\0\u01b0\0\u01c0\0\u01d0\0\u01e0\0\u01f0"+
    "\0\u0200\0\u0210\0\u0220\0\u0230\0\u0240\0\u0250\0\u0260\0\u0270"+
    "\0\u0280\0\320\0\u0290\0\u02a0\0\u02b0\0\u02c0\0\u02d0\0\u02e0"+
    "\0\u0150\0\u02f0\0\u0300\0\u0310\0\u0320\0\u0330\0\u0340\0\u0350"+
    "\0\u0360\0\u0370\0\u0380\0\u0390\0\240\0\u03a0\0\u03b0\0\u03c0"+
    "\0\u03d0\0\u03e0\0\u03f0\0\u0400\0\u0410\0\u0420\0\u0430\0\u0440"+
    "\0\u0450\0\u0460\0\u0470\0\u0480\0\u0490\0\u04a0\0\u04b0\0\u04c0"+
    "\0\u04d0\0\u04e0\0\u04f0\0\u02e0\0\u0150\0\u0500\0\u0510\0\u0520"+
    "\0\u0530\0\u0540\0\u0550\0\u0560\0\u0570\0\u0580\0\300\0\u01b0"+
    "\0\u0590\0\u05a0\0\u05b0\0\u05c0\0\u05d0\0\u05e0\0\u05f0\0\u0600"+
    "\0\u0610\0\u0620\0\u0630\0\u0640\0\u0650\0\u0660\0\u0670\0\u0680"+
    "\0\u0690\0\u06a0\0\u06b0\0\u0360\0\u06c0\0\u06d0\0\u06e0\0\u06f0"+
    "\0\u0700\0\u0710\0\u0720\0\u0730\0\u0740\0\u0750\0\u0760\0\u0770"+
    "\0\u0780\0\u0790\0\u07a0\0\u07b0\0\u07c0\0\u07d0\0\u07e0\0\u07f0"+
    "\0\u0800\0\u0810\0\u0820\0\u0830\0\u0840\0\u0850\0\u0860\0\u0870"+
    "\0\u0880\0\u0890\0\u08a0\0\u08b0\0\u08c0\0\u08d0\0\u08e0\0\u08f0"+
    "\0\u0900\0\u0910\0\u0920\0\u0930\0\u0940\0\u0950\0\u0960\0\u0970"+
    "\0\u0980\0\u0990\0\u09a0\0\u09b0\0\u09c0\0\u09d0\0\u09e0\0\u09f0"+
    "\0\u0a00\0\u0a10\0\u0a20\0\u0a30";

  private static int [] zzUnpackRowMap() {
    int [] result = new int[172];
    int offset = 0;
    offset = zzUnpackRowMap(ZZ_ROWMAP_PACKED_0, offset, result);
    return result;
  }

  private static int zzUnpackRowMap(String packed, int offset, int [] result) {
    int i = 0;  /* index in packed string  */
    int j = offset;  /* index in unpacked array */
    int l = packed.length();
    while (i < l) {
      int high = packed.charAt(i++) << 16;
      result[j++] = high | packed.charAt(i++);
    }
    return j;
  }

  /** 
   * The transition table of the DFA
   */
  private static final int [] ZZ_TRANS = zzUnpackTrans();

  private static final String ZZ_TRANS_PACKED_0 =
    "\2\2\1\3\5\2\1\4\1\5\1\6\1\7\1\10"+
    "\1\11\1\12\1\2\31\0\2\13\6\0\1\14\1\15"+
    "\1\16\1\17\2\20\1\21\1\22\1\23\1\24\1\7"+
    "\1\25\5\0\1\26\1\0\1\27\2\30\1\31\1\32"+
    "\2\23\1\7\1\33\4\0\1\14\1\34\1\16\1\17"+
    "\2\30\1\31\1\35\1\23\1\24\1\7\1\36\13\0"+
    "\3\37\2\7\4\0\1\14\1\15\1\16\1\17\2\20"+
    "\1\21\1\25\1\40\1\41\1\7\1\25\20\0\1\11"+
    "\21\0\1\2\2\0\1\42\5\0\1\43\2\44\1\7"+
    "\1\45\13\0\1\46\1\0\1\46\1\0\1\46\13\0"+
    "\1\47\1\50\1\51\1\0\1\47\13\0\1\52\1\0"+
    "\1\52\1\0\1\52\13\0\1\53\1\54\1\53\1\0"+
    "\1\53\13\0\1\55\2\56\1\0\1\55\13\0\1\21"+
    "\2\57\1\0\1\21\4\0\1\14\1\60\1\16\1\17"+
    "\2\20\1\21\1\22\1\23\1\24\1\7\1\25\5\0"+
    "\1\61\1\0\1\27\2\30\1\31\1\32\2\23\1\7"+
    "\1\33\4\0\1\14\1\61\1\16\1\17\2\30\1\31"+
    "\1\35\1\23\1\24\1\7\1\36\4\0\1\14\1\60"+
    "\1\16\1\17\2\20\1\21\1\25\1\40\1\41\1\7"+
    "\1\25\5\0\1\42\5\0\1\62\2\63\1\7\1\64"+
    "\13\0\3\54\1\0\1\54\13\0\1\65\2\66\1\0"+
    "\1\65\13\0\1\67\2\70\1\0\1\67\5\0\1\71"+
    "\1\0\1\27\2\30\1\31\1\32\2\23\1\7\1\33"+
    "\5\0\1\71\1\0\1\27\2\30\1\31\1\33\2\40"+
    "\1\7\1\33\5\0\1\42\5\0\1\72\1\63\1\73"+
    "\1\7\1\74\4\0\1\14\1\71\1\16\1\17\2\30"+
    "\1\31\1\35\1\23\1\24\1\7\1\36\4\0\1\14"+
    "\1\71\1\16\1\17\2\30\1\31\1\36\1\40\1\41"+
    "\1\7\1\36\13\0\3\37\1\0\1\37\5\0\1\61"+
    "\1\0\1\27\2\30\1\31\1\33\2\40\1\7\1\33"+
    "\4\0\1\14\1\61\1\16\1\17\2\30\1\31\1\36"+
    "\1\40\1\41\1\7\1\36\14\0\2\75\7\0\1\76"+
    "\2\0\3\76\1\43\2\44\1\7\1\45\5\0\1\75"+
    "\2\0\3\77\1\100\2\44\1\7\1\101\5\0\1\76"+
    "\2\0\3\76\1\45\2\102\1\7\1\45\4\0\1\14"+
    "\6\0\1\46\1\0\1\46\1\0\1\46\5\0\1\103"+
    "\1\0\1\27\2\104\1\0\1\105\2\50\1\0\1\105"+
    "\5\0\1\106\1\0\1\27\2\107\1\110\1\111\2\112"+
    "\1\7\1\113\5\0\1\114\1\0\1\27\2\107\1\110"+
    "\1\111\2\112\1\7\1\113\5\0\1\115\2\0\1\115"+
    "\2\0\1\53\1\54\1\53\1\0\1\53\5\0\1\115"+
    "\2\0\1\115\2\0\3\54\1\0\1\54\5\0\1\104"+
    "\1\0\1\27\2\104\1\0\1\55\2\56\1\0\1\55"+
    "\5\0\1\107\1\0\1\27\2\107\1\110\1\116\2\117"+
    "\1\7\1\120\5\0\1\110\2\0\3\110\1\121\2\122"+
    "\1\7\1\123\13\0\1\105\2\50\1\0\1\105\5\0"+
    "\1\124\1\0\1\27\2\20\1\21\1\62\2\63\1\7"+
    "\1\64\5\0\1\125\1\0\1\27\2\30\1\31\1\126"+
    "\2\63\1\7\1\127\5\0\1\124\1\0\1\27\2\20"+
    "\1\21\1\64\2\130\1\7\1\64\5\0\1\20\1\0"+
    "\1\27\2\20\1\21\1\65\2\131\1\7\1\132\5\0"+
    "\1\30\1\0\1\27\2\30\1\31\1\66\2\131\1\7"+
    "\1\133\5\0\1\21\2\0\3\21\1\67\2\134\1\7"+
    "\1\135\5\0\1\31\2\0\3\31\1\70\2\134\1\7"+
    "\1\136\13\0\1\62\2\126\1\0\1\62\5\0\1\137"+
    "\1\0\1\27\2\20\1\21\1\62\2\63\1\7\1\64"+
    "\5\0\1\140\1\0\1\27\2\30\1\31\1\126\2\63"+
    "\1\7\1\127\5\0\1\137\1\0\1\27\2\20\1\21"+
    "\1\64\2\130\1\7\1\64\13\0\1\76\2\141\1\0"+
    "\1\76\13\0\1\43\2\100\1\0\1\43\5\0\1\77"+
    "\2\0\3\77\1\100\2\44\1\7\1\101\5\0\1\77"+
    "\2\0\3\77\1\101\2\102\1\7\1\101\5\0\1\75"+
    "\2\0\3\77\1\101\2\102\1\7\1\101\13\0\1\142"+
    "\1\143\1\142\1\0\1\142\13\0\3\144\1\0\1\144"+
    "\5\0\1\145\1\0\1\27\2\104\1\0\1\105\2\50"+
    "\1\0\1\105\13\0\3\146\1\0\1\146\13\0\3\147"+
    "\1\0\1\147\13\0\3\150\1\0\1\150\5\0\1\151"+
    "\1\0\1\27\2\152\1\153\1\111\2\112\1\7\1\113"+
    "\5\0\1\154\1\0\1\27\2\152\1\153\1\111\2\112"+
    "\1\7\1\113\5\0\1\151\1\0\1\27\2\152\1\153"+
    "\1\113\2\155\1\7\1\113\13\0\1\156\1\146\1\156"+
    "\1\0\1\156\13\0\3\157\1\0\1\157\5\0\1\152"+
    "\1\0\1\27\2\152\1\153\1\116\2\117\1\7\1\120"+
    "\5\0\1\160\1\0\1\27\2\152\1\153\1\116\2\117"+
    "\1\7\1\120\5\0\1\152\1\0\1\27\2\152\1\153"+
    "\1\120\2\161\1\7\1\120\5\0\1\153\2\0\3\153"+
    "\1\121\2\122\1\7\1\123\5\0\1\162\2\0\3\153"+
    "\1\121\2\122\1\7\1\123\5\0\1\153\2\0\3\153"+
    "\1\123\2\163\1\7\1\123\5\0\1\164\1\0\1\27"+
    "\2\30\1\31\1\126\2\63\1\7\1\127\5\0\1\164"+
    "\1\0\1\27\2\30\1\31\1\127\2\130\1\7\1\127"+
    "\5\0\1\125\1\0\1\27\2\30\1\31\1\127\2\130"+
    "\1\7\1\127\5\0\1\165\1\0\1\27\2\30\1\31"+
    "\1\66\2\131\1\7\1\133\5\0\1\20\1\0\1\27"+
    "\2\20\1\21\1\132\2\166\1\7\1\132\5\0\1\30"+
    "\1\0\1\27\2\30\1\31\1\133\2\166\1\7\1\133"+
    "\5\0\1\167\2\0\3\31\1\70\2\134\1\7\1\136"+
    "\5\0\1\21\2\0\3\21\1\135\2\170\1\7\1\135"+
    "\5\0\1\31\2\0\3\31\1\136\2\170\1\7\1\136"+
    "\5\0\1\171\2\0\3\171\1\100\2\44\1\7\1\101"+
    "\5\0\1\103\1\0\1\27\2\104\1\0\3\143\1\0"+
    "\1\143\5\0\1\145\1\0\1\27\2\104\1\0\3\143"+
    "\1\0\1\143\5\0\1\104\1\0\1\27\2\104\1\0"+
    "\3\144\1\0\1\144\13\0\3\143\1\0\1\143\5\0"+
    "\1\124\1\0\1\27\2\20\1\21\1\146\2\172\1\7"+
    "\1\173\5\0\1\20\1\0\1\27\2\20\1\21\1\147"+
    "\2\174\1\7\1\175\5\0\1\21\2\0\3\21\1\150"+
    "\2\176\1\7\1\177\13\0\1\146\2\200\1\0\1\146"+
    "\13\0\1\147\2\201\1\0\1\147\13\0\1\150\2\202"+
    "\1\0\1\150\5\0\1\42\5\0\1\146\2\172\1\7"+
    "\1\173\5\0\1\154\1\0\1\27\2\152\1\153\1\113"+
    "\2\155\1\7\1\113\5\0\1\137\1\0\1\27\2\20"+
    "\1\21\1\146\2\172\1\7\1\173\5\0\1\115\2\0"+
    "\1\115\2\0\3\157\1\0\1\157\5\0\1\42\5\0"+
    "\1\147\2\174\1\7\1\175\5\0\1\160\1\0\1\27"+
    "\2\152\1\153\1\120\2\161\1\7\1\120\5\0\1\42"+
    "\5\0\1\150\2\176\1\7\1\177\5\0\1\162\2\0"+
    "\3\153\1\123\2\163\1\7\1\123\5\0\1\42\5\0"+
    "\1\65\2\131\1\7\1\132\5\0\1\165\1\0\1\27"+
    "\2\30\1\31\1\133\2\166\1\7\1\133\5\0\1\42"+
    "\5\0\1\67\2\134\1\7\1\135\5\0\1\167\2\0"+
    "\3\31\1\136\2\170\1\7\1\136\13\0\3\43\1\0"+
    "\1\43\5\0\1\203\1\0\1\27\2\204\1\205\1\200"+
    "\2\172\1\7\1\206\5\0\1\124\1\0\1\27\2\20"+
    "\1\21\1\173\2\207\1\7\1\173\5\0\1\210\1\0"+
    "\1\27\2\204\1\205\1\201\2\174\1\7\1\211\5\0"+
    "\1\20\1\0\1\27\2\20\1\21\1\175\2\212\1\7"+
    "\1\175\5\0\1\213\2\0\3\205\1\202\2\176\1\7"+
    "\1\214\5\0\1\21\2\0\3\21\1\177\2\215\1\7"+
    "\1\177\5\0\1\216\1\0\1\27\2\204\1\205\1\200"+
    "\2\172\1\7\1\206\5\0\1\204\1\0\1\27\2\204"+
    "\1\205\1\201\2\174\1\7\1\211\5\0\1\205\2\0"+
    "\3\205\1\202\2\176\1\7\1\214\5\0\1\42\5\0"+
    "\1\217\2\112\1\7\1\220\13\0\1\221\2\116\1\0"+
    "\1\221\13\0\1\222\2\121\1\0\1\222\5\0\1\216"+
    "\1\0\1\27\2\204\1\205\1\206\2\207\1\7\1\206"+
    "\5\0\1\203\1\0\1\27\2\204\1\205\1\206\2\207"+
    "\1\7\1\206\5\0\1\42\5\0\1\221\2\117\1\7"+
    "\1\223\5\0\1\204\1\0\1\27\2\204\1\205\1\211"+
    "\2\212\1\7\1\211\5\0\1\210\1\0\1\27\2\204"+
    "\1\205\1\211\2\212\1\7\1\211\5\0\1\42\5\0"+
    "\1\222\2\122\1\7\1\224\5\0\1\205\2\0\3\205"+
    "\1\214\2\215\1\7\1\214\5\0\1\213\2\0\3\205"+
    "\1\214\2\215\1\7\1\214\13\0\1\217\2\111\1\0"+
    "\1\217\5\0\1\225\1\0\1\27\2\226\1\76\1\217"+
    "\2\112\1\7\1\220\5\0\1\225\1\0\1\27\2\226"+
    "\1\76\1\220\2\155\1\7\1\220\5\0\1\226\1\0"+
    "\1\27\2\226\1\76\1\221\2\117\1\7\1\223\5\0"+
    "\1\76\2\0\3\76\1\222\2\122\1\7\1\224\5\0"+
    "\1\226\1\0\1\27\2\226\1\76\1\223\2\161\1\7"+
    "\1\223\5\0\1\76\2\0\3\76\1\224\2\163\1\7"+
    "\1\224\13\0\1\227\2\230\1\0\1\227\13\0\1\231"+
    "\2\232\1\0\1\231\5\0\1\145\1\0\1\27\2\104"+
    "\1\0\1\227\2\230\1\0\1\227\5\0\1\233\1\0"+
    "\1\27\2\234\1\171\1\235\2\236\1\7\1\237\5\0"+
    "\1\104\1\0\1\27\2\104\1\0\1\231\2\232\1\0"+
    "\1\231\5\0\1\234\1\0\1\27\2\234\1\171\1\240"+
    "\2\241\1\7\1\242\13\0\3\243\1\0\1\243\13\0"+
    "\3\244\1\0\1\244\5\0\1\245\1\0\1\27\2\246"+
    "\1\77\1\235\2\236\1\7\1\237\5\0\1\247\1\0"+
    "\1\27\2\246\1\77\1\235\2\236\1\7\1\237\5\0"+
    "\1\245\1\0\1\27\2\246\1\77\1\237\2\250\1\7"+
    "\1\237\5\0\1\246\1\0\1\27\2\246\1\77\1\240"+
    "\2\241\1\7\1\242\5\0\1\251\1\0\1\27\2\246"+
    "\1\77\1\240\2\241\1\7\1\242\5\0\1\246\1\0"+
    "\1\27\2\246\1\77\1\242\2\252\1\7\1\242\5\0"+
    "\1\225\1\0\1\27\2\226\1\76\1\243\2\236\1\7"+
    "\1\253\5\0\1\226\1\0\1\27\2\226\1\76\1\244"+
    "\2\241\1\7\1\254\13\0\1\243\2\235\1\0\1\243"+
    "\13\0\1\244\2\240\1\0\1\244\5\0\1\42\5\0"+
    "\1\243\2\236\1\7\1\253\5\0\1\247\1\0\1\27"+
    "\2\246\1\77\1\237\2\250\1\7\1\237\5\0\1\42"+
    "\5\0\1\244\2\241\1\7\1\254\5\0\1\251\1\0"+
    "\1\27\2\246\1\77\1\242\2\252\1\7\1\242\5\0"+
    "\1\225\1\0\1\27\2\226\1\76\1\253\2\250\1\7"+
    "\1\253\5\0\1\226\1\0\1\27\2\226\1\76\1\254"+
    "\2\252\1\7\1\254\3\0";

  private static int [] zzUnpackTrans() {
    int [] result = new int[2624];
    int offset = 0;
    offset = zzUnpackTrans(ZZ_TRANS_PACKED_0, offset, result);
    return result;
  }

  private static int zzUnpackTrans(String packed, int offset, int [] result) {
    int i = 0;       /* index in packed string  */
    int j = offset;  /* index in unpacked array */
    int l = packed.length();
    while (i < l) {
      int count = packed.charAt(i++);
      int value = packed.charAt(i++);
      value--;
      do result[j++] = value; while (--count > 0);
    }
    return j;
  }


  /* error codes */
  private static final int ZZ_UNKNOWN_ERROR = 0;
  private static final int ZZ_NO_MATCH = 1;
  private static final int ZZ_PUSHBACK_2BIG = 2;

  /* error messages for the codes above */
  private static final String ZZ_ERROR_MSG[] = {
    "Unkown internal scanner error",
    "Error: could not match input",
    "Error: pushback value was too large"
  };

  /**
   * ZZ_ATTRIBUTE[aState] contains the attributes of state <code>aState</code>
   */
  private static final int [] ZZ_ATTRIBUTE = zzUnpackAttribute();

  private static final String ZZ_ATTRIBUTE_PACKED_0 =
    "\1\0\1\11\11\1\6\0\5\1\3\0\10\1\3\0"+
    "\7\1\2\0\2\1\2\0\7\1\1\0\3\1\4\0"+
    "\3\1\1\0\2\1\2\0\4\1\1\0\23\1\1\0"+
    "\2\1\1\0\5\1\2\0\4\1\1\0\1\1\1\0"+
    "\2\1\1\0\1\1\1\0\1\1\1\0\12\1\2\0"+
    "\2\1\1\0\2\1\1\0\5\1\2\0\3\1\1\0"+
    "\2\1\2\0\1\1\1\0\3\1\2\0\2\1\1\0"+
    "\1\1\1\0\2\1\1\0\3\1";

  private static int [] zzUnpackAttribute() {
    int [] result = new int[172];
    int offset = 0;
    offset = zzUnpackAttribute(ZZ_ATTRIBUTE_PACKED_0, offset, result);
    return result;
  }

  private static int zzUnpackAttribute(String packed, int offset, int [] result) {
    int i = 0;       /* index in packed string  */
    int j = offset;  /* index in unpacked array */
    int l = packed.length();
    while (i < l) {
      int count = packed.charAt(i++);
      int value = packed.charAt(i++);
      do result[j++] = value; while (--count > 0);
    }
    return j;
  }

  /** the input device */
  private java.io.Reader zzReader;

  /** the current state of the DFA */
  private int zzState;

  /** the current lexical state */
  private int zzLexicalState = YYINITIAL;

  /** this buffer contains the current text to be matched and is
      the source of the yytext() string */
  private char zzBuffer[] = new char[ZZ_BUFFERSIZE];

  /** the textposition at the last accepting state */
  private int zzMarkedPos;

  /** the textposition at the last state to be included in yytext */
  private int zzPushbackPos;

  /** the current text position in the buffer */
  private int zzCurrentPos;

  /** startRead marks the beginning of the yytext() string in the buffer */
  private int zzStartRead;

  /** endRead marks the last character in the buffer, that has been read
      from input */
  private int zzEndRead;

  /** number of newlines encountered up to the start of the matched text */
  private int yyline;

  /** the number of characters up to the start of the matched text */
  private int yychar;

  /**
   * the number of characters from the last newline up to the start of the 
   * matched text
   */
  private int yycolumn;

  /** 
   * zzAtBOL == true <=> the scanner is currently at the beginning of a line
   */
  private boolean zzAtBOL = true;

  /** zzAtEOF == true <=> the scanner is at the EOF */
  private boolean zzAtEOF;

  /* user code: */

public static final int ALPHANUM          = 0;
public static final int APOSTROPHE        = 1;
public static final int ACRONYM           = 2;
public static final int COMPANY           = 3;
public static final int EMAIL             = 4;
public static final int HOST              = 5;
public static final int NUM               = 6;
public static final int CJ                = 7;
/**
 * @deprecated this solves a bug where HOSTs that end with '.' are identified
 *             as ACRONYMs. It is deprecated and will be removed in the next
 *             release.
 */
public static final int ACRONYM_DEP       = 8;
public static final int KOREAN            = 9;

public static final String [] TOKEN_TYPES = new String [] {
    "<ALPHANUM>",
    "<APOSTROPHE>",
    "<ACRONYM>",
    "<COMPANY>",
    "<EMAIL>",
    "<HOST>",
    "<NUM>", 
    "<CJ>",   
    "<ACRONYM_DEP>",
    "<KOREAN>"    
};

public final int yychar()
{
    return yychar;
}

/**
 * Fills Lucene token with the current token text.
 */
final void getText(Token t) {
  t.setTermBuffer(zzBuffer, zzStartRead, zzMarkedPos-zzStartRead);
}


/**
 * Fills CharTermAttribute with the current token text.
 */
public final void getText(CharTermAttribute t) {
  t.copyBuffer(zzBuffer, zzStartRead, zzMarkedPos-zzStartRead);
}


  /**
   * Creates a new scanner
   * There is also a java.io.InputStream version of this constructor.
   *
   * @param   in  the java.io.Reader to read input from.
   */
  KoreanTokenizerImpl(java.io.Reader in) {
    this.zzReader = in;
  }

  /**
   * Creates a new scanner.
   * There is also java.io.Reader version of this constructor.
   *
   * @param   in  the java.io.Inputstream to read input from.
   */
  KoreanTokenizerImpl(java.io.InputStream in) {
    this(new java.io.InputStreamReader(in));
  }

  /** 
   * Unpacks the compressed character translation table.
   *
   * @param packed   the packed character translation table
   * @return         the unpacked character translation table
   */
  private static char [] zzUnpackCMap(String packed) {
    char [] map = new char[0x10000];
    int i = 0;  /* index in packed string  */
    int j = 0;  /* index in unpacked array */
    while (i < 156) {
      int  count = packed.charAt(i++);
      char value = packed.charAt(i++);
      do map[j++] = value; while (--count > 0);
    }
    return map;
  }


  /**
   * Refills the input buffer.
   *
   * @return      <code>false</code>, iff there was new input.
   * 
   * @exception   java.io.IOException  if any I/O-Error occurs
   */
  private boolean zzRefill() throws java.io.IOException {

    /* first: make room (if you can) */
    if (zzStartRead > 0) {
      System.arraycopy(zzBuffer, zzStartRead,
                       zzBuffer, 0,
                       zzEndRead-zzStartRead);

      /* translate stored positions */
      zzEndRead-= zzStartRead;
      zzCurrentPos-= zzStartRead;
      zzMarkedPos-= zzStartRead;
      zzPushbackPos-= zzStartRead;
      zzStartRead = 0;
    }

    /* is the buffer big enough? */
    if (zzCurrentPos >= zzBuffer.length) {
      /* if not: blow it up */
      char newBuffer[] = new char[zzCurrentPos*2];
      System.arraycopy(zzBuffer, 0, newBuffer, 0, zzBuffer.length);
      zzBuffer = newBuffer;
    }

    /* finally: fill the buffer with new input */
    int numRead = zzReader.read(zzBuffer, zzEndRead,
                                            zzBuffer.length-zzEndRead);

    if (numRead < 0) {
      return true;
    }
    else {
      zzEndRead+= numRead;
      return false;
    }
  }

    
  /**
   * Closes the input stream.
   */
  public final void yyclose() throws java.io.IOException {
    zzAtEOF = true;            /* indicate end of file */
    zzEndRead = zzStartRead;  /* invalidate buffer    */

    if (zzReader != null)
      zzReader.close();
  }


  /**
   * Resets the scanner to read from a new input stream.
   * Does not close the old reader.
   *
   * All internal variables are reset, the old input stream 
   * <b>cannot</b> be reused (internal buffer is discarded and lost).
   * Lexical state is set to <tt>ZZ_INITIAL</tt>.
   *
   * @param reader   the new input stream 
   */
  public final void yyreset(java.io.Reader reader) {
    zzReader = reader;
    zzAtBOL  = true;
    zzAtEOF  = false;
    zzEndRead = zzStartRead = 0;
    zzCurrentPos = zzMarkedPos = zzPushbackPos = 0;
    yyline = yychar = yycolumn = 0;
    zzLexicalState = YYINITIAL;
  }


  /**
   * Returns the current lexical state.
   */
  public final int yystate() {
    return zzLexicalState;
  }


  /**
   * Enters a new lexical state
   *
   * @param newState the new lexical state
   */
  public final void yybegin(int newState) {
    zzLexicalState = newState;
  }


  /**
   * Returns the text matched by the current regular expression.
   */
  public final String yytext() {
    return new String( zzBuffer, zzStartRead, zzMarkedPos-zzStartRead );
  }


  /**
   * Returns the character at position <tt>pos</tt> from the 
   * matched text. 
   * 
   * It is equivalent to yytext().charAt(pos), but faster
   *
   * @param pos the position of the character to fetch. 
   *            A value from 0 to yylength()-1.
   *
   * @return the character at position pos
   */
  public final char yycharat(int pos) {
    return zzBuffer[zzStartRead+pos];
  }


  /**
   * Returns the length of the matched text region.
   */
  public final int yylength() {
    return zzMarkedPos-zzStartRead;
  }


  /**
   * Reports an error that occured while scanning.
   *
   * In a wellformed scanner (no or only correct usage of 
   * yypushback(int) and a match-all fallback rule) this method 
   * will only be called with things that "Can't Possibly Happen".
   * If this method is called, something is seriously wrong
   * (e.g. a JFlex bug producing a faulty scanner etc.).
   *
   * Usual syntax/scanner level error handling should be done
   * in error fallback rules.
   *
   * @param   errorCode  the code of the errormessage to display
   */
  private void zzScanError(int errorCode) {
    String message;
    try {
      message = ZZ_ERROR_MSG[errorCode];
    }
    catch (ArrayIndexOutOfBoundsException e) {
      message = ZZ_ERROR_MSG[ZZ_UNKNOWN_ERROR];
    }

    throw new Error(message);
  } 


  /**
   * Pushes the specified amount of characters back into the input stream.
   *
   * They will be read again by then next call of the scanning method
   *
   * @param number  the number of characters to be read again.
   *                This number must not be greater than yylength()!
   */
  public void yypushback(int number)  {
    if ( number > yylength() )
      zzScanError(ZZ_PUSHBACK_2BIG);

    zzMarkedPos -= number;
  }


  /**
   * Resumes scanning until the next regular expression is matched,
   * the end of input is encountered or an I/O-Error occurs.
   *
   * @return      the next token
   * @exception   java.io.IOException  if any I/O-Error occurs
   */
  public int getNextToken() throws java.io.IOException {
    int zzInput;
    int zzAction;

    // cached fields:
    int zzCurrentPosL;
    int zzMarkedPosL;
    int zzEndReadL = zzEndRead;
    char [] zzBufferL = zzBuffer;
    char [] zzCMapL = ZZ_CMAP;

    int [] zzTransL = ZZ_TRANS;
    int [] zzRowMapL = ZZ_ROWMAP;
    int [] zzAttrL = ZZ_ATTRIBUTE;

    while (true) {
      zzMarkedPosL = zzMarkedPos;

      yychar+= zzMarkedPosL-zzStartRead;

      zzAction = -1;

      zzCurrentPosL = zzCurrentPos = zzStartRead = zzMarkedPosL;
  
      zzState = zzLexicalState;


      zzForAction: {
        while (true) {
    
          if (zzCurrentPosL < zzEndReadL)
            zzInput = zzBufferL[zzCurrentPosL++];
          else if (zzAtEOF) {
            zzInput = YYEOF;
            break zzForAction;
          }
          else {
            // store back cached positions
            zzCurrentPos  = zzCurrentPosL;
            zzMarkedPos   = zzMarkedPosL;
            boolean eof = zzRefill();
            // get translated positions and possibly new buffer
            zzCurrentPosL  = zzCurrentPos;
            zzMarkedPosL   = zzMarkedPos;
            zzBufferL      = zzBuffer;
            zzEndReadL     = zzEndRead;
            if (eof) {
              zzInput = YYEOF;
              break zzForAction;
            }
            else {
              zzInput = zzBufferL[zzCurrentPosL++];
            }
          }
          int zzNext = zzTransL[ zzRowMapL[zzState] + zzCMapL[zzInput] ];
          if (zzNext == -1) break zzForAction;
          zzState = zzNext;

          int zzAttributes = zzAttrL[zzState];
          if ( (zzAttributes & 1) == 1 ) {
            zzAction = zzState;
            zzMarkedPosL = zzCurrentPosL;
            if ( (zzAttributes & 8) == 8 ) break zzForAction;
          }

        }
      }

      // store back cached position
      zzMarkedPos = zzMarkedPosL;

      switch (zzAction < 0 ? zzAction : ZZ_ACTION[zzAction]) {
        case 7: 
          { return HOST;
          }
        case 12: break;
        case 10: 
          { return ACRONYM_DEP;
          }
        case 13: break;
        case 9: 
          { return ACRONYM;
          }
        case 14: break;
        case 1: 
          { /* ignore */
          }
        case 15: break;
        case 5: 
          { return NUM;
          }
        case 16: break;
        case 4: 
          { return CJ;
          }
        case 17: break;
        case 2: 
          { return ALPHANUM;
          }
        case 18: break;
        case 8: 
          { return COMPANY;
          }
        case 19: break;
        case 6: 
          { return APOSTROPHE;
          }
        case 20: break;
        case 3: 
          { return KOREAN;
          }
        case 21: break;
        case 11: 
          { return EMAIL;
          }
        case 22: break;
        default: 
          if (zzInput == YYEOF && zzStartRead == zzCurrentPos) {
            zzAtEOF = true;
            return YYEOF;
          } 
          else {
            zzScanError(ZZ_NO_MATCH);
          }
      }
    }
  }


}
